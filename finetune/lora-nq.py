"""
Instruction-tuning with LoRA on the Alpaca dataset.

Note: If you run into a CUDA error "Expected is_sm80 to be true, but got false", uncomment the line
`torch.backends.cuda.enable_flash_sdp(False)` in the script below (see https://github.com/Lightning-AI/lit-llama/issues/101).
"""
import sys
from pathlib import Path
import os
import time

import lightning as L
import numpy as np
import torch
import math

from tqdm import *
from metrics import *

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from generate import generate
from lit_llama.lora import mark_only_lora_as_trainable, lora, lora_state_dict
from lit_llama.model import LLaMA, LLaMAConfig
from lit_llama.tokenizer import Tokenizer
from scripts.prepare_alpaca import generate_prompt


instruction_tuning = True
eval_interval = 100
eval_iters = 300
log_interval = 10
devices = torch.cuda.device_count()
# Hyperparameters
learning_rate = 1e-4
init_lr = 1e-6
batch_size = 128
micro_batch_size = 1
gradient_accumulation_iters = batch_size // micro_batch_size
assert gradient_accumulation_iters > 0
weight_decay = 0.0
max_seq_length = 1024  # see scripts/prepare_alpaca.py
lora_r = 16
lora_alpha = 16
lora_dropout = 0.05
warmup_iters = 50



def main(
    data_dir: str = "data/alpaca", 
    pretrained_path: str = "/home/jskai/workspace/models/lit-llama/7B/lit-llama.pth",
    tokenizer_path: str = "/home/jskai/workspace/models/lit-llama/tokenizer.model",
    out_dir: str = "out/lora/alpaca",
    resume_path: str = None,
):

    fabric = L.Fabric(accelerator="cuda", devices=devices, strategy="ddp", precision="bf16-true")
    fabric.launch()
    fabric.seed_everything(608 + fabric.global_rank)

    if fabric.global_rank == 0:
        os.makedirs(out_dir, exist_ok=True)

    train_data, val_data, test_data = load_datasets(data_dir=data_dir)

    config = LLaMAConfig.from_name("7B")
    config.block_size = max_seq_length

    checkpoint = torch.load(pretrained_path)

    with fabric.init_module(), lora(r=lora_r, alpha=lora_alpha, dropout=lora_dropout, enabled=True):
        model = LLaMA(config)
        # strict=False because missing keys due to LoRA weights not contained in checkpoint state
        if resume_path is not None and os.path.isfile(resume_path):
            print(F"resuming LORA from {resume_path}...")
            lora_ckpt = torch.load(resume_path)
            checkpoint.update(lora_ckpt)
        model.load_state_dict(checkpoint, strict=False)
    
    mark_only_lora_as_trainable(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    model, optimizer = fabric.setup(model, optimizer)
    train(fabric, model, optimizer, train_data, val_data, tokenizer_path, out_dir)

    fabric.barrier()
    scores = test(fabric, model, test_data, tokenizer_path)
    fabric.print(scores)
    fabric.barrier()

    # Save the final LoRA checkpoint at the end of training
    checkpoint = lora_state_dict(model)
    fabric.save(os.path.join(out_dir, "lit-llama-lora-finetuned.pth"), checkpoint)


def train(
    fabric: L.Fabric,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    train_data: np.ndarray,
    val_data: np.ndarray,
    tokenizer_path: str,
    out_dir: str,
) -> None:
    """The training loop.

    Loosely based on the nanoGPT implementation: https://github.com/karpathy/nanoGPT.
    """
    init_iter = 0
    step_count = 0
    for param_group in optimizer.param_groups:
        param_group['lr'] = init_lr
    fabric.print("-"*50)
    fabric.print(f"lr {init_lr:.6f}")

    for iter_num in tqdm(range(len(train_data) * 3 // micro_batch_size)):
        t0 = time.time()

        input_ids, targets1, targets2 = get_batch(fabric, train_data)
        if iter_num < init_iter:
            if (iter_num + 1) % gradient_accumulation_iters == 0:
                optimizer.step()
                optimizer.zero_grad()
                step_count += 1
                fabric.barrier()
            continue
        with fabric.no_backward_sync(model, enabled=((iter_num + 1) % gradient_accumulation_iters != 0)):
            logits1 = model(torch.cat((input_ids, targets1), dim=1))[:, input_ids.shape[1]:]
            loss1 = loss_fn(logits1, targets1)
            fabric.backward(loss1 / gradient_accumulation_iters)
            
            # logits2 = model(torch.cat((input_ids, targets2), dim=1))[:, input_ids.shape[1]:]
            # loss2 = loss_fn(logits2, targets2)
            # fabric.backward(loss2 / gradient_accumulation_iters)
            fabric.barrier()

        if (iter_num + 1) % gradient_accumulation_iters == 0:
            optimizer.step()
            optimizer.zero_grad()
            step_count += 1
            torch.cuda.empty_cache()
            if step_count <= warmup_iters:
                # linear warmup
                lr = learning_rate * step_count / warmup_iters
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr
            else:
                # inverse-sqrt decay
                lr = learning_rate * math.sqrt(warmup_iters / step_count)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr
            fabric.print("-"*50)
            fabric.print(f"step {iter_num}: lr {lr:.6f}")
                
            if step_count % eval_interval == 0:
                val_loss, scores = validate(fabric, model, val_data, tokenizer_path)
                fabric.print("="*50)
                fabric.print(f"step {iter_num}: lr {lr:.6f}, val loss {val_loss:.4f}")
                fabric.print(f"f1-score: {scores:.4f}")
                fabric.barrier()
                torch.cuda.empty_cache()

                print(f"Saving LoRA weights to {out_dir}")
                # We are only saving the LoRA weights
                # TODO: Provide a function/script to merge the LoRA weights with pretrained weights
                checkpoint = lora_state_dict(model)
                fabric.save(os.path.join(out_dir, f"iter-{iter_num:06d}-ckpt.pth"), checkpoint)

        dt = time.time() - t0
        if iter_num % log_interval == 0:
            fabric.print(f"iter {iter_num}: loss {loss1.item():.4f}, time: {dt*1000:.2f}ms")


@torch.no_grad()
def validate(fabric: L.Fabric, model: torch.nn.Module, val_data: np.ndarray, tokenizer_path: str) -> torch.Tensor:
    fabric.print("Validating ...")
    model.eval()
    losses = torch.zeros(eval_iters)

    tokenizer = Tokenizer(tokenizer_path)
    f1_scores = []

    for k in tqdm(range(eval_iters)):
        input_ids, targets1, targets2 = get_batch(fabric, val_data)
        logits = model(torch.cat((input_ids, targets1), dim=1))[:, input_ids.shape[1]:]
        loss = loss_fn(logits, targets1)
        losses[k] = loss.item()
        
        output = generate(
            model,
            idx=input_ids.squeeze(0),
            max_seq_length=max_seq_length,
            max_new_tokens=16,
        )
        outputs = tokenizer.decode(output[input_ids.shape[1]:])
        stop_words = [".", "Question"]
        for stop_word in stop_words:
            outputs_tmp = outputs.split(stop_word)
            if outputs_tmp[0] != "":
                outputs = outputs_tmp[0]

        labels1 = tokenizer.decode(targets1[0])
        labels2 = tokenizer.decode(targets2[0])
        torch.cuda.empty_cache()
        f1_scores.append(max(qa_f1_score(outputs, labels1), qa_f1_score(outputs, labels2)))
        fabric.print(f"f1-score: {qa_f1_score(outputs, labels1):.4f}, {qa_f1_score(outputs, labels2):.4f}")
        fabric.print(f"outputs: {outputs}\n labels1: {labels1}\n labels2: {labels2}")
        # import pdb
        # pdb.set_trace()

    out = losses.mean()

    score = np.array(f1_scores).mean()

    model.train()
    return out.item(), score


@torch.no_grad()
def test(fabric: L.Fabric, model: torch.nn.Module, test_data: np.ndarray, tokenizer_path: str):
    fabric.print("Testing ...")
    model.eval()

    tokenizer = Tokenizer(tokenizer_path)
    f1_scores = []

    for k in tqdm(range(int(len(test_data) * 1))):    
        input_ids, targets1, targets2 = get_batch(fabric, test_data)
        output = generate(
            model,
            idx=input_ids.squeeze(0),
            max_seq_length=max_seq_length,
            max_new_tokens=16,
        )
        outputs = tokenizer.decode(output[input_ids.shape[1]:])
        stop_words = [".", "Question"]
        for stop_word in stop_words:
            outputs_tmp = outputs.split(stop_word)
            if outputs_tmp[0] != "":
                outputs = outputs_tmp[0]
                
        labels1 = tokenizer.decode(targets1[0])
        labels2 = tokenizer.decode(targets2[0])
        torch.cuda.empty_cache()
        f1_scores.append(max(qa_f1_score(outputs, labels1), qa_f1_score(outputs, labels2)))

    score = np.array(f1_scores).mean()

    model.train()
    return score


def loss_fn(logits, targets):
    # shift the targets such that output n predicts token n+1
    logits = logits[..., :-1, :].contiguous()
    targets = targets[..., 1:].contiguous()
    loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
    return loss
    

def get_batch(fabric: L.Fabric, data: list):
    ix = torch.randint(len(data), (micro_batch_size,))

    input_ids = [data[i]["input_ids"].type(torch.int64) for i in ix]
    labels1 = [data[i]["labels1"].type(torch.int64) for i in ix]
    labels2 = [data[i]["labels2"].type(torch.int64) for i in ix]

    input_max_len = max(len(s) for s in input_ids)
    label_max_len1 = max(len(s) for s in labels1)
    label_max_len2 = max(len(s) for s in labels2)

    def pad_right(x, max_len, pad_id):
        # pad right based on the longest sequence
        n = max_len - len(x)
        return torch.cat((x, torch.full((n,), pad_id, dtype=x.dtype)))

    x = torch.stack([pad_right(x, max_len=input_max_len, pad_id=0) for x in input_ids])[:, :min(input_max_len, 1024 - 16)]
    y1 = torch.stack([pad_right(x, max_len=label_max_len1, pad_id=-1) for x in labels1])[:, :min(label_max_len1, 16)]
    y2 = torch.stack([pad_right(x, max_len=label_max_len2, pad_id=-1) for x in labels2])[:, :min(label_max_len2, 16)]
    x, y1, y2 = fabric.to_device((x.pin_memory(), y1.pin_memory(), y2.pin_memory()))
    return x, y1, y2


def load_datasets(data_dir):
    train_data = torch.load(os.path.join(data_dir, "train.pt"))
    val_data = torch.load(os.path.join(data_dir, "validation.pt"))
    test_data = torch.load(os.path.join(data_dir, "test.pt"))
    return train_data, val_data, test_data


if __name__ == "__main__":
    # Uncomment this line if you see an error: "Expected is_sm80 to be true, but got false"
    # torch.backends.cuda.enable_flash_sdp(False)
    torch.set_float32_matmul_precision("high")
    
    from jsonargparse.cli import CLI

    CLI(main)
