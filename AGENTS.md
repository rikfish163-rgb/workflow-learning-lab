# Workflow Learning Lab

Research objective: learn from verified workflow recovery and measure transfer to
first attempts on unseen scenarios. A functioning pipeline is not a positive
experimental result. Keep measured, proposed, and unverified claims separate.

Commands: `python3 -m pytest`; `python3 -m ruff check .`.

- Never execute model-generated Python in the host process. AppWorld execution
  uses the constrained Docker worker. Keep upstream safety checks enabled.
- No test feedback, task solutions, evaluation assertions, or verifier internals
  may enter the policy, experience writer, training data, or retrieval index.
- A recovered suffix is not a successful first attempt. Report both separately.
- Split by scenario before any data mining. Freeze evaluation IDs and seeds.
- Paired branches start from a fresh environment and replay the same prefix.
  Check observation equality; database restore alone loses interpreter state.
- Count failed runs, replay overhead, generation, and validation cost. Transport
  failures are infrastructure errors, not model failures or successful examples.
- CPU test doubles verify the harness only. Never present their scores as LLM results.
- Do not publish AppWorld protected data, tasks, traces or derived solution text.
- Keep optional model/training dependencies separate from the stdlib core.
