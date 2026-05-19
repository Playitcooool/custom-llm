# Teachers Guide for Codex

This folder owns teacher abstractions for distillation.

## Big Picture

- `base.py` defines the `Teacher` interface with a single `answer(prompt: str) -> str` method.
- `mock.py` implements a deterministic offline teacher used by the default distillation command and tests.

## Change Guidance

- Keep `MockTeacher` deterministic and dependency-free.
- Put API-backed or model-backed teachers behind optional dependencies and avoid importing those clients at package import time.
- Do not change `Teacher.answer` casually; `train/distill.py` depends on this minimal interface.
- If adding a real teacher, make failure modes explicit and keep tests able to run offline.

