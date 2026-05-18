from custom_llm.teachers.base import Teacher


class MockTeacher(Teacher):
    def answer(self, prompt: str) -> str:
        words = prompt.strip().split()
        tail = " ".join(reversed(words[-8:]))
        return f"mock teacher: {tail}" if tail else "mock teacher: empty prompt"
