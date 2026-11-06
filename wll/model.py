"""Bounded client for an explicit loopback inference endpoint."""
import json
import urllib.error
import urllib.request
from urllib.parse import urlparse


class ContextBudgetExceeded(RuntimeError):
    pass


class LocalModel:
    def __init__(self, endpoint, model_id, *, max_tokens=512, temperature=.7, timeout=180):
        parsed = urlparse(endpoint)
        if parsed.scheme != "http" or parsed.hostname not in {"localhost", "127.0.0.1", "::1"}:
            raise ValueError("use an SSH tunnel to a loopback http endpoint")
        if parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError("endpoint cannot contain credentials, queries or fragments")
        self.endpoint = endpoint.rstrip("/")
        self.model_id = model_id
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout = timeout
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def generate(self, messages, seed):
        payload = {"model": self.model_id, "messages": messages, "seed": seed,
                   "max_tokens": self.max_tokens, "temperature": self.temperature}
        request = urllib.request.Request(self.endpoint + "/chat/completions",
                                         json.dumps(payload).encode(),
                                         headers={"Content-Type": "application/json"})
        try:
            with self.opener.open(request, timeout=self.timeout) as response:
                raw = response.read(4_000_001)
        except urllib.error.HTTPError as exc:
            detail = exc.read(4096)
            if exc.code == 400 and b"context_budget_exceeded" in detail:
                raise ContextBudgetExceeded() from exc
            raise
        if len(raw) > 4_000_000:
            raise RuntimeError("oversized model response")
        result = json.loads(raw)
        return {"text": result["choices"][0]["message"]["content"],
                "prompt_tokens": result["usage"]["prompt_tokens"],
                "completion_tokens": result["usage"]["completion_tokens"],
                "format_error": result.get("format_error"),
                "backend": result.get("backend")}
