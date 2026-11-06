import pytest

from wll.code_policy import violation


@pytest.mark.parametrize("code", [
    "import appworld", "from appworld.task import Task", "import os",
    "getattr(apis, '__dict__')", "print(apis.__dict__)",
    "x = __import__('appworld')", "print(globals())", "print(vars(apis))",
    "import importlib", "print(apis.spotify.login.__globals__)",
    "f = eval; f('1')", "class Evil: pass", "apis.spotify.login = print",
    "del apis.spotify.login", "print(type(apis))", "import pickle",
])
def test_forbid_internal_access_and_reflection(code):
    assert violation(code) is not None


@pytest.mark.parametrize("code", [
    "import json\nprint(json.dumps({'a':1}))",
    "from datetime import datetime\nprint(datetime.now())",
    "response = apis.api_docs.show_app_descriptions()\nprint(response)",
    "rows = [r for r in response if r['amount'] > 3]\nprint(len(rows))",
    "for page in range(1, 5):\n    print(page)",
    "def add(a,b):\n    return a+b\nprint(add(1,2))",
    "response['key'] = 1",
])
def test_allow_normal_workflow_operations(code):
    assert violation(code) is None


def test_syntax_errors_remain_visible_environment_feedback():
    assert violation("for") is None
