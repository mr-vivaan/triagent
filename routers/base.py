from abc import ABC, abstractmethod
from schemas import RouteDecision


class Router(ABC):
    name = "base"

    @abstractmethod
    def route(self, text: str) -> RouteDecision: ...
