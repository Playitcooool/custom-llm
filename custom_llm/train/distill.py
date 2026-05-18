import json
from pathlib import Path

from custom_llm.data.datasets import chat_template
from custom_llm.teachers.mock import MockTeacher
from custom_llm.train.sft import run_sft


def run_distill(config: dict, prompts_file: str, tokenizer_path: str, out: str | None = None):
    teacher = MockTeacher()
    tmp = Path(".distill_teacher_data.jsonl")
    rows = []
    for prompt in Path(prompts_file).read_text().splitlines():
        if prompt.strip():
            rows.append({"prompt": prompt, "response": teacher.answer(chat_template(prompt))})
    tmp.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    try:
        return run_sft(config, str(tmp), tokenizer_path, out)
    finally:
        tmp.unlink(missing_ok=True)
