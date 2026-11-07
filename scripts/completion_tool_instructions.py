"""Fixed shared tool instructions for both completion-check conditions.

This text describes the existing protocol and five-endpoint registry. It does not
dispatch tools, select a check, supply task evidence or authorize a study.
"""

_TOOL_INSTRUCTIONS = """The pending completion action has not executed and has no observation. Use only the task and already-visible history. These tool rules are identical for both check conditions.

INSPECTION TURN
Return exactly one JSON object with exactly the keys "check" and "tool". Do not output Python, markdown fences, extra fields or multiple objects. "check" must be a nonblank string of at most 2048 characters. The whole request must be at most 32768 UTF-8 bytes.

At most one inspection query is permitted in total. "tool" is either null or one object with exactly "app", "api" and "arguments". No lists of tools, batches or follow-up inspection requests. tool:null means no inspection query and no new inspection evidence; it is not a successful verification. Invalid requests are rejected, without automatic repair or resampling.

Each arguments value must use exactly one of these forms:
Literal form: {"literal": "a JSON value"}
Variable form: {"variable": "an_existing_actor_variable"}
Path form: {"variable": "an_existing_actor_variable", "path": [0, "field"]}
These argument-value forms belong inside a complete request. A path uses dictionary string keys or nonnegative list indices only, with at most 16 parts. Keys must not start with an underscore and must have at most 256 characters; list indices must be integers in 0..1000000, not booleans. No attributes, expressions, slices or negative indices. Referenced values must be plain built-in JSON values, not objects, functions, modules or API handles. A variable name must match [A-Za-z][A-Za-z0-9_]{0,63}, have actually been assigned in visible actor history, and be available in the permitted namespace. Do not invent a missing name or value.

Only these five inspection endpoints are available. The argument sets are exact:
- api_docs.show_app_descriptions: arguments {} only, with no parameters.
- api_docs.show_api_descriptions: exactly app_name.
- api_docs.show_api_doc: exactly app_name and api_name.
- gmail.show_email: exactly email_id and access_token.
- file_system.show_file: exactly file_path and access_token.

Public app_name values: amazon, api_docs, file_system, gmail, phone, simple_note, splitwise, spotify, supervisor, todoist, venmo
app_name is case-sensitive and must be one of those values; admin and every unlisted app are forbidden. The public app_name enum is for documentation arguments, not permission to call other inspection endpoints. show_api_doc only retrieves documentation for the named API; it does not invoke that API.
api_name character length: 1..64; spelling must match [A-Za-z][A-Za-z0-9_]{0,63}.
email_id integer range: 1..9223372036854775807; JSON true/false are not integers.
access_token character length: 1..8192; use an already-known token for the corresponding app, not a guessed token.
file_path character length: 1..4096; this is a simulated file_system app path subject to its normal path and account-authorization checks, not a host path.
Gmail requires sender/recipient authorization; file_system requires account ownership. Inspection cannot log in, refresh credentials, initialize resources or call a mutating endpoint. A request or returned error is not proof that the requested state holds.

Never use framework objects as JSON variable references. Forbidden framework/interpreter references include apis, requester, request_tracker, ApiCollection, client, shell, user_ns, get_ipython, builtins, globals and locals. In later Python, apis is usable only for the direct endpoint calls described below; never alias, rebind, subscript, mutate or pass around apis, its app containers or endpoint functions. No eval, exec, reflection or private names. Inspection requests are resolved by the trusted dispatcher; they do not execute Python.

The following six complete JSON examples show syntax only, not which check to choose. Each is an independent alternative. The variable names are placeholders: use them only if the actual visible history assigned and exposed them. The "check" text is also a placeholder, not evidence or an answer.
{"check":"State your check.","tool":{"app":"api_docs","api":"show_app_descriptions","arguments":{}}}
{"check":"State your check.","tool":{"app":"api_docs","api":"show_api_descriptions","arguments":{"app_name":{"literal":"gmail"}}}}
{"check":"State your check.","tool":{"app":"api_docs","api":"show_api_doc","arguments":{"app_name":{"literal":"file_system"},"api_name":{"literal":"show_file"}}}}
{"check":"State your check.","tool":{"app":"gmail","api":"show_email","arguments":{"email_id":{"variable":"email_id"},"access_token":{"variable":"gmail_token"}}}}
{"check":"State your check.","tool":{"app":"file_system","api":"show_file","arguments":{"file_path":{"variable":"files","path":[0,"path"]},"access_token":{"variable":"accounts","path":["file_system","access_token"]}}}}
{"check":"State your check.","tool":null}

AFTER INSPECTION
Wait for the actual inspection observation or explicit skipped result. Then return to the ordinary Python correction/completion phase under the existing code-output rules and remaining budget. That phase may use public APIs for reads or justified changes; the five-endpoint list above limits inspection only. In Python, apis may appear only in direct calls of the form apis.<app>.<api>(...). Keep all existing code restrictions, including the ban on framework access and API aliases. Do not fabricate an inspection result, a future completion observation or a task outcome.
"""


def get_tool_instructions() -> str:
    """Return the same fixed tool instructions; condition prompts are separate."""
    return _TOOL_INSTRUCTIONS
