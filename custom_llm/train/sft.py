import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from custom_llm.data.collate import causal_lm_collate
from custom_llm.data.datasets import SFTDataset
from custom_llm.data.tokenizer import load_tokenizer
from custom_llm.model.checkpoints import save_checkpoint
from custom_llm.model.model import TinyGemmaLM
from custom_llm.train.utils import optimization_step, resolve_device, tiny_config_from_dict, train_cfg


def run_sft(
    config: dict,
    jsonl: str,
    tokenizer_path: str,
    out: str | None = None,
    device_name: str = "auto",
) -> TinyGemmaLM:
    cfg = tiny_config_from_dict(config)
    tcfg = train_cfg(config)
    dev = resolve_device(device_name)
    print(f"device: {dev}")
    tokenizer = load_tokenizer(tokenizer_path)
    cfg.vocab_size = max(cfg.vocab_size, tokenizer.get_vocab_size())
    ds = SFTDataset(jsonl, tokenizer, tcfg.get("seq_len", min(128, cfg.max_seq_len)))
    dl = DataLoader(ds, batch_size=tcfg.get("batch_size", 2), shuffle=True, collate_fn=causal_lm_collate)
    model = TinyGemmaLM(cfg).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=tcfg.get("lr", 3e-4))
    it = iter(dl)
    for _ in tqdm(range(tcfg.get("steps", 10)), desc="sft"):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(dl)
            batch = next(it)
        optimization_step(model, batch, opt, dev)
    if out:
        save_checkpoint(model.cpu(), out)
    return model
