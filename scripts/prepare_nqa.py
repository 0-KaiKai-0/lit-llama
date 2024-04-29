"""Implementation derived from https://github.com/tloen/alpaca-lora"""
import sys
from pathlib import Path

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

import torch
import requests
import json
from torch.utils.data import random_split
from lit_llama.tokenizer import Tokenizer
from tqdm import tqdm
from datasets import load_dataset


def prepare(
    destination_path: Path = Path("data/narrativeqa16"),
    tokenizer_path: Path = Path("/home/jskai/workspace/models/lit-llama/tokenizer.model"),
    max_seq_length: int = 1024,
) -> None:
    """Prepare any dataset for finetuning (akin to Shakespheare full tuning).

    The output is a training and validation dataset saved as `train.pt` and `val.pt`,
    which stores the preprocessed and tokenized prompts and labels.
    """

    destination_path.mkdir(parents=True, exist_ok=True)
    
    raw_datasets = load_dataset(
        'narrativeqa',
        cache_dir=None,
        use_auth_token=None,
    )
    splits = ["validation", "test", "train"]
    column_names = raw_datasets["train"].column_names
    text_column = column_names[0]
    question_column = column_names[1]
    answer_column = column_names[2]

    tokenizer = Tokenizer(tokenizer_path)

    for split in splits:
        if split not in raw_datasets:
            raise ValueError(split + " dataset required")
        dataset = raw_datasets[split]

        print(f"{split} has {len(dataset):,} samples")
        print(f"Processing {split} split ...")
        dataset = [
            prepare_line(data[text_column]['summary']['text'], 
                            data[question_column]['text'], 
                            data[answer_column][0]['text'], 
                            data[answer_column][1]['text'], 
                            tokenizer, max_seq_length) for data in tqdm(dataset)
        ]
        torch.save(dataset, destination_path / f"{split}.pt")


def prepare_line(inputs1: str, inputs2: str, targets1: str, targets2: str, tokenizer: Tokenizer, max_length: int):
    """Processes a single sample.

    This function processes the line to produce the tokenized version of it.
    """
    prompt = (
            "You are given a story, which can be either a novel or a movie script, and a question."
            "Answer the question asconcisely as you can.\n\n"
            f"Story: {inputs1}\n\n"
        )
    appendix = f"Now, answer the question based on the story asconcisely as you can, using a single phrase if possible. Do not provide any explanation.\n\nQuestion: {inputs2}\n\nAnswer: "
    appendix_ids = tokenizer.encode(appendix, max_length=max_length)
    prompt_ids = tokenizer.encode(prompt, max_length=max_length - appendix_ids.shape[0] - 16)
    input_ids = torch.cat((prompt_ids, appendix_ids), dim=0)
    labels1 = tokenizer.encode(targets1, max_length=max_length)
    labels2 = tokenizer.encode(targets2, max_length=max_length)
    # import pdb
    # pdb.set_trace()
    return {
        "input_ids": input_ids,
        "labels1": labels1,
        "labels2": labels2,
    }


if __name__ == "__main__":
    from jsonargparse import CLI

    CLI(prepare)
