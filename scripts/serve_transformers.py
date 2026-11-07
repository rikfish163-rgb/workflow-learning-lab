"""Single-GPU pilot server. Local weights, loopback HTTP, no generated code execution."""
import argparse
import hashlib
import importlib.metadata
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


def generation_options(*, temperature, max_new, pad_token_id, top_p, top_k=None, min_p=None):
    if not 0 <= temperature <= 2 or max_new < 1 or not 0 < top_p <= 1 or (
            top_k is not None and top_k < 0) or (min_p is not None and not 0 <= min_p <= 1):
        raise ValueError("invalid_generation_parameters")
    options = {"do_sample": temperature > 0, "max_new_tokens": max_new,
               "pad_token_id": pad_token_id}
    if temperature > 0:
        options.update(temperature=temperature, top_p=top_p)
        if top_k is not None:
            options["top_k"] = top_k
        if min_p is not None:
            options["min_p"] = min_p
    return options


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--port", type=int, default=18876)
    parser.add_argument("--max-context", type=int, default=16384)
    parser.add_argument("--loader", choices=["causal", "multimodal"], default="causal")
    parser.add_argument("--adapter-path", help="Local PEFT adapter produced by this project")
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument("--max-output-tokens", type=int, default=2048)
    parser.add_argument("--top-p", type=float, default=.95)
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--min-p", type=float)
    args = parser.parse_args()
    generation_options(temperature=.7, max_new=args.max_output_tokens, pad_token_id=0,
                       top_p=args.top_p, top_k=args.top_k, min_p=args.min_p)

    from wll.gpu_policy import acquire_gpu

    with acquire_gpu() as gpu:
        serve(args, gpu)


def serve(args, gpu):
    import torch
    from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer

    from wll.gpu_policy import verify_cuda

    verify_cuda(torch)
    path = Path(args.model_path).resolve(strict=True)
    backend = {"server_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               "gpu_policy": gpu,
               "packages": {n: importlib.metadata.version(n) for n in ["torch", "transformers"]},
               "files": {n: hashlib.sha256((path / n).read_bytes()).hexdigest() for n in
                         ["config.json", "tokenizer_config.json", "generation_config.json", "weight-source.json"]
                         if (path / n).is_file()},
               "top_p": args.top_p, "top_k": args.top_k, "min_p": args.min_p,
               "thinking": args.enable_thinking, "max_context": args.max_context,
               "max_output_tokens": args.max_output_tokens, "loader": args.loader}
    if args.adapter_path:
        adapter = Path(args.adapter_path).resolve(strict=True)
        backend["adapter_files"] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                    for p in adapter.iterdir() if p.suffix in {".json", ".safetensors"}}
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=False)
    model_class = AutoModelForCausalLM if args.loader == "causal" else AutoModelForImageTextToText
    model = model_class.from_pretrained(path, local_files_only=True,
                                       trust_remote_code=False, dtype=torch.bfloat16)
    if args.adapter_path:
        from peft import PeftModel

        adapter = Path(args.adapter_path).resolve(strict=True)
        if not (adapter / "adapter_model.safetensors").is_file():
            raise ValueError("adapter must contain local safetensors weights")
        model = PeftModel.from_pretrained(model, adapter, local_files_only=True,
                                         is_trainable=False)
    model.to("cuda").eval()
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def setup(self):
            super().setup()
            self.connection.settimeout(180)

        def log_message(self, format, *values):
            pass

        def respond(self, status, body):
            payload = json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def do_GET(self):
            if self.path == "/health":
                self.respond(200, {"model": args.model_id, "ready": True,
                                   "torch": torch.__version__, "adapter": args.adapter_path,
                                   "max_context": args.max_context, "loader": args.loader,
                                   "thinking": args.enable_thinking,
                                   "max_output_tokens": args.max_output_tokens,
                                   "backend": backend})
            else:
                self.respond(404, {"error": "not_found"})

        def do_POST(self):
            if self.path != "/v1/chat/completions":
                self.respond(404, {"error": "not_found"})
                return
            try:
                length = int(self.headers.get("Content-Length", 0))
                if not 0 < length <= 2_000_000:
                    raise ValueError("invalid_request_size")
                request = json.loads(self.rfile.read(length))
                if request["model"] != args.model_id:
                    raise ValueError("wrong_model")
                messages = request["messages"]
                if not messages or any(m["role"] not in {"system", "user", "assistant"}
                                       or not isinstance(m["content"], str) for m in messages):
                    raise ValueError("invalid_messages")
                max_new = int(request.get("max_tokens", 512))
                temperature = float(request.get("temperature", .7))
                seed = int(request.get("seed", 0))
                if not 1 <= max_new <= args.max_output_tokens or not 0 <= temperature <= 2:
                    raise ValueError("invalid_generation_parameters")
                with lock, torch.inference_mode():
                    prompt = tokenizer.apply_chat_template(messages, tokenize=False,
                                                           add_generation_prompt=True,
                                                           enable_thinking=args.enable_thinking)
                    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
                    n = inputs.input_ids.shape[1]
                    if n + max_new > args.max_context:
                        raise ValueError("context_budget_exceeded")
                    torch.manual_seed(seed)
                    options = generation_options(temperature=temperature, max_new=max_new,
                        pad_token_id=tokenizer.eos_token_id, top_p=args.top_p,
                        top_k=args.top_k, min_p=args.min_p)
                    outputs = model.generate(**inputs, **options)
                    tokens = outputs[0, n:]
                    text = tokenizer.decode(tokens, skip_special_tokens=True)
                    format_error = None
                    if args.enable_thinking:
                        if "</think>" in text:
                            text = text.rsplit("</think>", 1)[1].strip()
                        else:
                            text = ""
                            format_error = "reasoning budget exhausted before final answer"
                    self.respond(200, {"model": args.model_id,
                        "choices": [{"message": {"role": "assistant", "content": text}}],
                        "format_error": format_error,
                        "backend": backend,
                        "usage": {"prompt_tokens": n, "completion_tokens": len(tokens)}})
            except ValueError as exc:
                self.respond(400, {"error": str(exc)})
            except Exception as exc:
                self.respond(500, {"error": type(exc).__name__})

    print(json.dumps({"ready": True, "model": args.model_id, "port": args.port}), flush=True)
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
