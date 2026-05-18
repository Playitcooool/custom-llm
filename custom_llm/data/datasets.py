import json
from pathlib import Path

import torch
from torch.utils.data import Dataset

from custom_llm.data.tokenizer import encode


def pack_sequences(ids: list[int], seq_len: int) -> list[list[int]]:
    usable = len(ids) - (len(ids) % seq_len)
    return [ids[i : i + seq_len] for i in range(0, usable, seq_len)]


class PackedTextDataset(Dataset):
    def __init__(self, text_files: list[str], tokenizer, seq_len: int):
        ids = []
        for file in text_files:
            ids.extend(encode(tokenizer, Path(file).read_text()))
        self.blocks = pack_sequences(ids, seq_len)

    def __len__(self):
        return len(self.blocks)

    def __getitem__(self, idx):
        x = torch.tensor(self.blocks[idx], dtype=torch.long)
        return {"input_ids": x, "labels": x.clone()}


def chat_template(prompt: str, response: str | None = None) -> str:
    text = f"<user>\n{prompt}\n<assistant>\n"
    if response is not None:
        text += response
    return text


class SFTDataset(Dataset):
    def __init__(self, jsonl_file: str, tokenizer, seq_len: int):
        self.rows = []
        for line in Path(jsonl_file).read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            prompt_ids = encode(tokenizer, chat_template(row["prompt"]), add_eos=False)
            response_ids = encode(tokenizer, row["response"], add_bos=False)
            if len(prompt_ids) + len(response_ids) > seq_len:
                response_ids = response_ids[:seq_len]
                prompt_budget = max(0, seq_len - len(response_ids))
                prompt_ids = prompt_ids[-prompt_budget:] if prompt_budget else []
            ids = (prompt_ids + response_ids)[:seq_len]
            labels = ids.copy()
            for i in range(min(len(prompt_ids), len(labels))):
                labels[i] = -100
            self.rows.append((ids, labels))

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        ids, labels = self.rows[idx]
        return {"input_ids": torch.tensor(ids), "labels": torch.tensor(labels)}
