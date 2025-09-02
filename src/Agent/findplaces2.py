import ollama
from smolagents import CodeAgent

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

# usage
model = OllamaModel("qwen2:7b")
agent = CodeAgent(tools=[], model=model)
print(agent.run("Plan a 1-day Paris bicycle itinerary with 5–6 stops and times."))
