import torch


def causal_lm_collate(rows: list[dict], pad_id: int = 0) -> dict[str, torch.Tensor]:
    max_len = max(len(row["input_ids"]) for row in rows)
    input_ids, labels = [], []
    for row in rows:
        x = row["input_ids"]
        y = row["labels"]
        pad = max_len - len(x)
        input_ids.append(torch.cat([x, torch.full((pad,), pad_id, dtype=torch.long)]))
        labels.append(torch.cat([y, torch.full((pad,), -100, dtype=torch.long)]))
    return {"input_ids": torch.stack(input_ids), "labels": torch.stack(labels)}
