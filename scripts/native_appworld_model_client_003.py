"""Typed count propagation over the authenticated frozen NativeModel client.

HTTP, backend/usage validation and single-use generation binding are inherited
unchanged. Count errors retain allowlisted private audit metadata and causes.
The custom one-argument counter remains an explicit synthetic-test seam.
"""

import math
import time
from pathlib import Path

from scripts import native_appworld_model_client as previous
from scripts import native_verification_token_counter_003 as counter

NativeModelError = previous.NativeModelError
ActorCountDeadlineExpired = counter.ActorCountDeadlineExpired
BASE_SOURCE_SHA256 = "cada4da0696a4f7b4e138edd54beb0bb62fd5db0e439a6b0586d14cb404e0344"
BASE_COUNTER_SHA256 = "da28d8639fbd3a003284e4126925cc0fe65bf68ae4d9c5c26238a90adc94c8d7"


def _verify_sources():
    if (counter._source_digest(previous.__file__) != BASE_SOURCE_SHA256
            or counter._source_digest(Path(previous.__file__).with_name("native_verification_token_counter.py"))
            != BASE_COUNTER_SHA256):
        raise ValueError("native_model_source_binding")
    counter._verify_protocol_source()


def _now(clock):
    value = clock()
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValueError("native_model_count_clock")
    return float(value)


class NativeModel(previous.NativeModel):
    def __init__(self, endpoint, model_id, *, expected_backend, max_tokens, context_tokens,
                 temperature, timeout, deadline, _token_counter=None, _opener=None, _clock=time.monotonic):
        try:
            _verify_sources()
        except Exception as error:
            raise NativeModelError("native_model_source_binding_failed") from error
        super().__init__(endpoint, model_id, expected_backend=expected_backend, max_tokens=max_tokens,
                         context_tokens=context_tokens, temperature=temperature, timeout=timeout,
                         deadline=deadline, _token_counter=_token_counter, _opener=_opener, _clock=_clock)
        self._counter = counter.count_input_tokens if _token_counter is None else _token_counter

    def count_input_tokens(self, messages):
        self._counted = None
        audit = counter._new_audit(self.deadline, counter.TIMEOUT_SECONDS)
        try:
            _verify_sources()
            if audit["actor_deadline"] is None:
                raise ValueError("native_model_count_deadline")
            audit["started_monotonic"] = _now(self._clock)
            payload, digest = previous._message_payload(messages)
            snapshot = previous._json(payload)["messages"]
            audit["messages_sha256"] = digest
            current = _now(self._clock)
            if current >= float(self.deadline):
                audit.update(finished_monotonic=current, deadline_limited=True)
                raise ActorCountDeadlineExpired(deadline=float(self.deadline), phase="pre_count", count_audit=audit)
            if self._default_counter:
                count = self._counter(snapshot, deadline=float(self.deadline), timeout_seconds=counter.TIMEOUT_SECONDS)
            else:
                audit["child_started"] = None
                count = self._counter(snapshot)
            _, after = previous._message_payload(snapshot)
            if type(count) is not int or count < 1 or after != digest:
                raise ValueError("native_model_count_binding")
            _verify_sources()
            finished = _now(self._clock)
            if finished < audit["started_monotonic"]:
                raise ValueError("native_model_count_clock")
            audit["finished_monotonic"] = finished
            if self._default_counter:
                audit.update(child_started=True, child_returncode=0, proof_validated=True)
            if finished >= float(self.deadline):
                if not audit["proof_validated"]:
                    raise ValueError("native_model_unproven_late_test_count")
                raise ActorCountDeadlineExpired(deadline=float(self.deadline), phase="validated_count_returned_late", count_audit=audit)
            self._counted = (digest, count)
            return count
        except ActorCountDeadlineExpired as error:
            try:
                _verify_sources()
                copied = counter.validate_count_audit(error.count_audit)
                if (type(error.deadline) is not float or error.deadline != float(self.deadline)
                        or error.phase not in counter._PHASES or copied["actor_deadline"] != error.deadline
                        or copied["messages_sha256"] != digest
                        or previous._message_payload(snapshot)[1] != digest
                        or copied["finished_monotonic"] is None or copied["finished_monotonic"] < error.deadline
                        or _now(self._clock) < error.deadline):
                    raise ValueError("native_model_unproven_count_deadline")
            except Exception as invalid:
                failure = NativeModelError("native_model_CPU_count_failed")
                failure.count_audit = counter.validate_count_audit(audit)
                raise failure from invalid
            error.count_audit = copied
            raise
        except Exception as error:
            try:
                audit["finished_monotonic"] = _now(self._clock)
                if audit["started_monotonic"] is not None and audit["finished_monotonic"] < audit["started_monotonic"]:
                    audit["finished_monotonic"] = None
            except Exception:
                pass
            try:
                copied = counter.validate_count_audit(error.count_audit)
            except Exception:
                copied = counter.validate_count_audit(audit)
            failure = NativeModelError("native_model_CPU_count_failed")
            failure.count_audit = copied
            raise failure from error
