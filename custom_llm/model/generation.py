import torch
import torch.nn.functional as F


@torch.no_grad()
def generate(
    model,
    input_ids: torch.Tensor,
    max_new_tokens: int = 32,
    temperature: float = 0.8,
    top_k: int = 50,
    eos_id: int | None = None,
    suppress_token_ids: list[int] | None = None,
    repetition_penalty: float = 1.0,
):
    if repetition_penalty < 1.0:
        raise ValueError("repetition_penalty must be >= 1.0")
    model.eval()
    suppress = None
    if suppress_token_ids:
        suppress = torch.tensor(suppress_token_ids, device=input_ids.device, dtype=torch.long)
    for _ in range(max_new_tokens):
        ctx = input_ids[:, -model.cfg.max_seq_len :]
        logits = model(ctx)["logits"][:, -1, :]
        if suppress is not None:
            logits[:, suppress] = -float("inf")
        if repetition_penalty > 1.0:
            seen = torch.zeros_like(logits, dtype=torch.bool)
            seen.scatter_(1, input_ids, True)
            penalized = torch.where(logits < 0, logits * repetition_penalty, logits / repetition_penalty)
            logits = torch.where(seen, penalized, logits)
        if temperature <= 0:
            next_id = logits.argmax(dim=-1, keepdim=True)
        else:
            logits = logits / temperature
            if top_k:
                vals, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits = logits.masked_fill(logits < vals[:, [-1]], -float("inf"))
            next_id = torch.multinomial(F.softmax(logits, dim=-1), 1)
        input_ids = torch.cat([input_ids, next_id], dim=1)
        if eos_id is not None and bool((next_id == eos_id).all()):
            break
    return input_ids
