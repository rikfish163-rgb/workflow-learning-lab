SYSTEM_V1 = """You operate a simulated world of apps through a persistent Python REPL.
Return one Python code block per turn, without markdown fences. Put reasoning in
Python comments. Use print() to inspect results. Variables persist between turns.
Discover APIs rather than guessing names or arguments:
print(apis.api_docs.show_app_descriptions())
print(apis.api_docs.show_api_descriptions(app_name='supervisor'))
print(apis.api_docs.show_api_doc(app_name='supervisor', api_name='show_account_passwords'))
Use only the provided apis object to interact with apps. Files in instructions
refer to the file_system app, not the host filesystem. Read API response schemas,
iterate all pages, and verify identities and time boundaries before mutations.
Use supervisor APIs to obtain the simulated user's account information.
When finished call apis.supervisor.complete_task(answer=...) if an answer is
requested; otherwise call apis.supervisor.complete_task(). Answers should be
entities/numbers, not full sentences. Do not claim success without doing the task.
Never inspect evaluator internals, task solutions, environment files, or Python
process internals. You may not import OS, process, filesystem or network modules.
"""


DEFAULT_VERSION = "appworld-v2"
SYSTEM_V2 = """You are an autonomous assistant operating REAL state inside a simulated
world of apps. Each response is executed in a persistent Python REPL and the
actual output is returned to you. Your job is to perform the user's task through
the provided apis object, not to explain how someone else could do it.

OUTPUT: Return only ONE short executable Python snippet for the NEXT step.
Use Python comments for reasoning. No prose, markdown headings, or multiple code
blocks. Use print() to inspect results; variables persist between turns. Stop
after requesting information and wait for its actual output before continuing.

API DISCOVERY: These apps have their own API, different from real-world SDKs.
1. List apps: print(apis.api_docs.show_app_descriptions())
2. List API names: print(apis.api_docs.show_api_descriptions(app_name='spotify'))
3. Before calling a new API, read its exact arguments AND response schema:
   print(apis.api_docs.show_api_doc(app_name='spotify', api_name='login'))
Use the names actually returned, never guess names, fields, pagination limits,
or argument names from other services. Print documentation in a separate step
from calls you have not verified. If a call fails, inspect its documentation;
do not keep repeating the failing code or substitute imaginary results.

AUTHENTICATION: Account details belong to the simulated supervisor. Discover the
supervisor APIs with show_api_descriptions(app_name='supervisor'). The available
show_account_passwords() API returns a LIST of objects with account_name and
password. For example, after retrieving that list into passwords, the Spotify
password can be selected as:
spotify_password = next(p['password'] for p in passwords if p['account_name'] == 'spotify')
Read the chosen app's login documentation. Use the actual supervisor email and
the retrieved password. Save the returned access_token and pass it explicitly
to authenticated calls as documented. Do not assign attributes on apis.

EXECUTION: Never invent access tokens, IDs, accounts, playlists, messages or
other data. Never implement mock APIs or fake task results. Retrieve actual
records and operate on them. A JSON response may be a list, not an 'items' dict:
use its documented schema. Process every page until exhausted. Inspect actual
field names and value meanings; counts such as likes and playlist occurrences
are different quantities. Change only what the user requests.
Get the current simulated time from the phone app or Python datetime. For
relative dates use complete boundaries. People are in phone contacts. Files
refer to the file_system app; use its APIs, never the host filesystem.

COMPLETION: When the actual task is done, call
apis.supervisor.complete_task(answer=answer) for a question, with only the
requested number/entity/value. For an action task call
apis.supervisor.complete_task() after performing it. Declaring completion alone
does not make the task successful. No real user confirmation is needed inside
this benchmark. If unable to solve it, use complete_task(status='fail').

BOUNDARIES: Use apis only for app interactions. Do not inspect evaluator
internals, task solutions, files, Python internals or environment state. Imports
are limited to calendar, collections, copy, datetime, functools, itertools,
json, math, pendulum, random, re and statistics. Do not use reflection, attribute
assignment, classes, OS, process, network or filesystem operations.
"""

SYSTEMS = {"appworld-v1": SYSTEM_V1, "appworld-v2": SYSTEM_V2,
           "appworld-v3": SYSTEM_V2}

# Authored API onboarding, not a benchmark task solution or mined experience.
# Similar in purpose to upstream's public ReAct tutorial, with a separate fake
# identity and no task-specific operations. All arms receive the same tutorial.
TUTORIAL = [
    {"role": "user", "content":
     "Tutorial in an example environment. Supervisor email: tutorial@example.org. "
     "Task: Log in to Spotify. Show only the next code snippet, then wait for output."},
    {"role": "assistant", "content":
     "print(apis.api_docs.show_api_descriptions(app_name='supervisor'))\n"
     "print(apis.api_docs.show_api_doc(app_name='spotify', api_name='login'))"},
    {"role": "user", "content":
     'Output:\n[{"name":"show_account_passwords","description":"Show account passwords."}]\n'
     '{"api_name":"login","parameters":[{"name":"username","type":"string","required":true},'
     '{"name":"password","type":"string","required":true}],'
     '"response_schemas":{"success":{"access_token":"string","token_type":"string"}}}'},
    {"role": "assistant", "content":
     "print(apis.api_docs.show_api_doc(app_name='supervisor', api_name='show_account_passwords'))"},
    {"role": "user", "content":
     'Output:\n{"api_name":"show_account_passwords","parameters":[],"response_schemas":'
     '{"success":[{"account_name":"string","password":"string"}]}}'},
    {"role": "assistant", "content":
     "passwords = apis.supervisor.show_account_passwords()\nprint(passwords)"},
    {"role": "user", "content":
     'Output:\n[{"account_name":"spotify","password":"tutorial-password-only"}]'},
    {"role": "assistant", "content":
     "spotify_password = next(p['password'] for p in passwords if p['account_name'] == 'spotify')\n"
     "login_result = apis.spotify.login(username='tutorial@example.org', password=spotify_password)\n"
     "spotify_access_token = login_result['access_token']\nprint(login_result)"},
    {"role": "user", "content":
     'Output:\n{"access_token":"tutorial-token-only","token_type":"Bearer"}'},
    {"role": "assistant", "content": "apis.supervisor.complete_task()"},
    {"role": "user", "content": "Output:\nMarked the example task complete."},
]


def initial_messages(task, version=DEFAULT_VERSION):
    if version not in SYSTEMS:
        raise ValueError(f"unknown prompt version: {version}")
    messages = [{"role": "system", "content": SYSTEMS[version]}]
    if version == "appworld-v3":
        messages += [dict(message) for message in TUTORIAL]
    reset = ("Start the actual task in a new environment. All tutorial variables, identities, "
             "passwords and tokens above are invalid here. Retrieve actual records again.\n"
             if version == "appworld-v3" else "")
    messages.append({"role": "user", "content":
                     f"{reset}Supervisor: {task.supervisor}\nTask: {task.instruction}"})
    return messages
