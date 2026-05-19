# Data Guide for Codex

This folder owns tokenizer training/loading, text-to-token dataset construction, SFT example formatting, collation, and bounded dataset preparation.

## Big Picture

- `tokenizer.py` trains and loads a byte-level BPE tokenizer with the repo's special tokens: `<pad>`, `<bos>`, `<eos>`, `<unk>`, `<user>`, `<assistant>`.
- `datasets.py` turns plain text into fixed-length causal LM blocks and JSONL `{prompt, response}` rows into supervised examples.
- `collate.py` pads `input_ids` with pad id `0` and masks padded labels with `-100`.
- `tinystories.py` and `fineweb_edu.py` prepare bounded text samples for local training.

## Important Contracts

- `encode` adds BOS and EOS by default. Callers opt out for prompt prefixes or response continuations.
- SFT prompt tokens must be masked with `-100` so loss is only taken on assistant response tokens.
- `PackedTextDataset` drops incomplete tail tokens when packing fixed-size blocks.
- Data preparation functions should write bounded outputs and create parent directories.
- Keep online dataset imports lazy, as in `prepare_fineweb_edu`, so the package can import without optional remote setup.

## Testing

Run data and training coverage after changes here:

```bash
rtk uv run pytest tests/test_data_and_train.py
```

Add tests when changing tokenizer special tokens, prompt formatting, masking, truncation, padding, or bounded sample-writing behavior.

