import ollama
import os, requests
from typing import Optional
from ddgs import DDGS
from smolagents import CodeAgent, HfApiModel, tool

# Try to use smolagents' own ChatMessage type if available; else fallback.
try:
    from smolagents.models import ChatMessage  # newer versions
except Exception:
    try:
        from smolagents import ChatMessage     # some versions export here
    except Exception:
        from dataclasses import dataclass
        @dataclass
        class ChatMessage:
            role: str
            content: str

def _stringify_content(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for p in content:
            if isinstance(p, dict):
                if p.get("type") == "text" and "text" in p:
                    parts.append(str(p["text"]))
                elif "content" in p:
                    parts.append(str(p["content"]))
            else:
                parts.append(str(p))
        return "\n".join(parts)
    return str(content)

class OllamaModel:
    def __init__(self, model_id: str):
        self.model_id = model_id

    def _normalize_messages(self, maybe_messages_or_prompt):
        if isinstance(maybe_messages_or_prompt, list):
            msgs = maybe_messages_or_prompt
        else:
            msgs = [{"role": "user", "content": str(maybe_messages_or_prompt)}]

        out = []
        for m in msgs:
            role = str(m.get("role", "user")).split(".")[-1].lower()
            out.append({"role": role, "content": _stringify_content(m.get("content", ""))})
        return out

    # smolagents may call this:
    def chat(self, messages, **gen_kwargs):
        msgs = self._normalize_messages(messages)
        res = ollama.chat(
            model=self.model_id,
            messages=msgs,
            options={k: v for k, v in gen_kwargs.items() if k in {"temperature", "top_p", "top_k", "num_predict"}},
        )
        text = res["message"]["content"]
        return ChatMessage(role="assistant", content=text)

    # smolagents may call the model like a function:
    def __call__(self, prompt=None, **gen_kwargs):
        msgs = self._normalize_messages(prompt)
        res = ollama.chat(
            model=self.model_id,
            messages=msgs,
            options={k: v for k, v in gen_kwargs.items() if k in {"temperature", "top_p", "top_k", "num_predict"}},
        )
        text = res["message"]["content"]
        return ChatMessage(role="assistant", content=text)

@tool
def duckduckgo_search(query: str, num_results: int = 5) -> str:
    """
    Search DuckDuckGo for a query and extract the most relevant information.
    Search for temperature in Fahrenheit or Celsius.
    If it is in Celsius, convert to Fahrenheit.

    Args:
        query: The search query string.
        num_results: Number of results to return (default 5).

    Result: Give temperature as {Temp}°F or {Temp}°C.
    """
    with DDGS() as ddgs:
        results = ddgs.text(query, max_results=num_results)
        output = []
        for r in results:
            title = r.get("title")
            link = r.get("href")
            snippet = r.get("body")
            output.append(f"{title} ({link}) - {snippet}")
        return "\n".join(output) if output else "No results found."

# usage
model = OllamaModel("qwen2:7b")
agent = CodeAgent(tools=[duckduckgo_search], model=model)
print(agent.run("I need to find the current temperature in Champaign"))  # example usage
