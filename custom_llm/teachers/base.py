from abc import ABC, abstractmethod


class Teacher(ABC):
    @abstractmethod
    def answer(self, prompt: str) -> str:
        raise NotImplementedError
