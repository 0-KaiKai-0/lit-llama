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
    destination_path: Path = Path("data/cnn-dm-heading96"),
    tokenizer_path: Path = Path("/home/jskai/workspace/models/lit-llama/tokenizer.model"),
    max_seq_length: int = 1024,
) -> None:
    """Prepare any dataset for finetuning (akin to Shakespheare full tuning).

    The output is a training and validation dataset saved as `train.pt` and `val.pt`,
    which stores the preprocessed and tokenized prompts and labels.
    """

    destination_path.mkdir(parents=True, exist_ok=True)
    
    raw_datasets = load_dataset(
        'cnn_dailymail',
        '3.0.0',
        cache_dir=None,
        use_auth_token=None,
    )
    splits = ["validation", "test", "train"]
    column_names = raw_datasets["train"].column_names
    text_column = column_names[0]
    summary_column = column_names[1]

    tokenizer = Tokenizer(tokenizer_path)

    for split in splits:
        if split not in raw_datasets:
            raise ValueError(split + " dataset required")
        dataset = raw_datasets[split]

        print(f"{split} has {len(dataset):,} samples")

        print(f"Processing {split} split ...")
        dataset = [
            prepare_line(data[text_column], data[summary_column], tokenizer, max_seq_length) for data in tqdm(dataset)
        ]
        torch.save(dataset, destination_path / f"{split}.pt")


def prepare_line(inputs: str, targets: str, tokenizer: Tokenizer, max_length: int):
    """Processes a single sample.

    This function processes the line to produce the tokenized version of it.
    """
    # inputs = "Below is an article of news stories. Write an abstractive summary for the following articles.\n\n" + inputs
    prompt = f"Below is an article of news stories. Write an abstractive summary for it.\n\n### Article:\n{inputs}\n\n"
    appendix = "### Summary:\n"
    # import pdb
    # pdb.set_trace()
    appendix_ids = tokenizer.encode(appendix, max_length=max_length)
    labels = tokenizer.encode(targets, max_length=max_length)
    prompt_ids = tokenizer.encode(prompt, max_length=max_length - appendix_ids.shape[0] - max(labels.shape[0], 96))
    input_ids = torch.cat((prompt_ids, appendix_ids), dim=0)
    return {
        "input_ids": input_ids,
        "labels": labels,
    }


if __name__ == "__main__":
    from jsonargparse import CLI

    CLI(prepare)
