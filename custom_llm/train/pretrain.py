import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from custom_llm.data.collate import causal_lm_collate
from custom_llm.data.datasets import PackedTextDataset
from custom_llm.data.tokenizer import load_tokenizer
from custom_llm.model.checkpoints import load_checkpoint, save_checkpoint
from custom_llm.model.model import TinyGemmaLM
from custom_llm.train.utils import optimization_step, resolve_device, tiny_config_from_dict, train_cfg


def run_pretrain(
    config: dict,
    text_files: list[str],
    tokenizer_path: str,
    out: str | None = None,
    device_name: str = "auto",
    restart_checkpoint: str | None = None,
) -> TinyGemmaLM:
    cfg = tiny_config_from_dict(config)
    tcfg = train_cfg(config)
    dev = resolve_device(device_name)
    print(f"device: {dev}")
    tokenizer = load_tokenizer(tokenizer_path)
    cfg.vocab_size = max(cfg.vocab_size, tokenizer.get_vocab_size())
    ds = PackedTextDataset(text_files, tokenizer, tcfg.get("seq_len", min(128, cfg.max_seq_len)))
    dl = DataLoader(ds, batch_size=tcfg.get("batch_size", 2), shuffle=True, collate_fn=causal_lm_collate)
    model = load_checkpoint(restart_checkpoint, cfg) if restart_checkpoint else TinyGemmaLM(cfg)
    model = model.to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=tcfg.get("lr", 3e-4))
    steps = tcfg.get("steps", 10)
    it = iter(dl)
    for _ in tqdm(range(steps), desc="pretrain"):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(dl)
            batch = next(it)
        optimization_step(model, batch, opt, dev)
    if out:
        save_checkpoint(model.cpu(), out)
    return model
