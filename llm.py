"""Pluggable LLM backend for the specialist agents. Pick with LLM=mock|gemini|ollama."""
import requests
from config import env


class MockLLM:
    """Offline stand-in so the whole pipeline runs with no keys (for tests/dev)."""
    def generate(self, system: str, prompt: str) -> str:
        policy = ""
        for line in prompt.splitlines():
            if line.startswith("- "):
                policy = line[2:]
                break
        return ("Hi, thanks for reaching out. I'm sorry for the trouble. "
                f"Here is what I can tell you: {policy or 'I will look into this for you.'} "
                "Let me know if you need anything else.")


class GeminiLLM:
    def __init__(self):
        self.key = env("GEMINI_API_KEY")
        self.model = env("GEMINI_MODEL", "gemini-2.0-flash")
        if not self.key:
            raise RuntimeError("Set GEMINI_API_KEY")

    def generate(self, system, prompt):
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent"
        body = {"system_instruction": {"parts": [{"text": system}]},
                "contents": [{"parts": [{"text": prompt}]}]}
        r = requests.post(url, params={"key": self.key}, json=body, timeout=60)
        r.raise_for_status()
        return r.json()["candidates"][0]["content"]["parts"][0]["text"].strip()


class OllamaLLM:
    def __init__(self):
        self.model = env("OLLAMA_MODEL", "llama3.2")

    def generate(self, system, prompt):
        r = requests.post("http://localhost:11434/api/chat", timeout=120, json={
            "model": self.model, "stream": False,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": prompt}]})
        r.raise_for_status()
        return r.json()["message"]["content"].strip()


def get_llm(name=None):
    name = (name or env("LLM", "mock")).lower()
    return {"mock": MockLLM, "gemini": GeminiLLM, "ollama": OllamaLLM}[name]()
