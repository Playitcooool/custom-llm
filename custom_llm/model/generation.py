import torch
import torch.nn.functional as F


@torch.no_grad()
def generate(model, input_ids: torch.Tensor, max_new_tokens: int = 32, temperature: float = 0.8, top_k: int = 50):
    model.eval()
    for _ in range(max_new_tokens):
        ctx = input_ids[:, -model.cfg.max_seq_len :]
        logits = model(ctx)["logits"][:, -1, :]
        if temperature <= 0:
            next_id = logits.argmax(dim=-1, keepdim=True)
        else:
            logits = logits / temperature
            if top_k:
                vals, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits = logits.masked_fill(logits < vals[:, [-1]], -float("inf"))
            next_id = torch.multinomial(F.softmax(logits, dim=-1), 1)
        input_ids = torch.cat([input_ids, next_id], dim=1)
    return input_ids
