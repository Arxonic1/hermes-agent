"""ASH-1785 — restricted zero-tool transcript-analysis lane (direct-client design).

After two security-review passes the lane makes ONE bounded chat-completions call
through a directly-constructed OpenAI-compatible client (no AIAgent lifecycle, and NOT
auxiliary_client.call_llm — whose auto-resolution/fallback could re-route the untrusted
transcript to the ambient main provider / a copilot-acp subprocess).

These tests prove: unsafe/underspecified runtimes are rejected before any call; SSRF
targets are blocked; the call carries zero tools + a bounded request; output is rebuilt
typed + capped with strict top-level validation; raw output is bounded before parsing;
and every failure fails closed with a CONSTANT (untainted) message.
"""

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from agent import restricted_transcript_lane as lane
from agent.restricted_transcript_lane import (
    RestrictedLaneViolation,
    analyze_transcript,
    sanitize_output,
)

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures" / "transcripts"

# Public IP literal → _assert_public_https takes the no-DNS path.
RUNTIME = {"provider": "openrouter", "model": "test/model",
           "base_url": "https://1.1.1.1/v1", "api_key": "test-key"}


def _load(name: str) -> str:
    return (FIXTURES / name).read_text()


def _resp(content):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


def _fake_client(content=None, *, raises=None, capture=None):
    def create(**kwargs):
        if capture is not None:
            capture.update(kwargs)
        if raises is not None:
            raise raises
        return _resp(content)
    return SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create)),
        close=lambda: None,
    )


def _patch_client(content=None, *, raises=None, capture=None):
    # Patch the lane's client-builder seam so no real httpx/openai client is constructed.
    return patch.object(lane, "_build_client",
                        return_value=_fake_client(content, raises=raises, capture=capture))


# --------------------------------------------------------------------------- #
# Runtime validation — reject host-capable / underspecified / SSRF runtimes
# --------------------------------------------------------------------------- #

class TestRuntimeValidation:
    def test_rejects_codex_app_server_api_mode(self):
        with pytest.raises(RestrictedLaneViolation):
            analyze_transcript("hi", runtime={**RUNTIME, "api_mode": "codex_app_server"})

    def test_rejects_unknown_api_mode(self):
        with pytest.raises(RestrictedLaneViolation):
            analyze_transcript("hi", runtime={**RUNTIME, "api_mode": "anthropic_messages"})

    def test_rejects_acp_provider(self):
        with pytest.raises(RestrictedLaneViolation):
            analyze_transcript("hi", runtime={**RUNTIME, "provider": "copilot-acp"})

    def test_requires_model_api_key_base_url(self):
        for missing in ("model", "api_key", "base_url"):
            rt = {**RUNTIME}
            del rt[missing]
            with pytest.raises(RestrictedLaneViolation):
                analyze_transcript("hi", runtime=rt)

    def test_rejects_non_string_fields(self):
        with pytest.raises(RestrictedLaneViolation):
            analyze_transcript("hi", runtime={**RUNTIME, "model": 123})

    def test_rejects_empty_or_whitespace_required_fields(self):
        for bad in ("", "   "):
            with pytest.raises(RestrictedLaneViolation):
                analyze_transcript("hi", runtime={**RUNTIME, "model": bad})

    @pytest.mark.parametrize("bad", [
        "acp://local", "http://1.1.1.1/v1",                 # plain http now rejected
        "http://127.0.0.1/v1", "https://192.168.1.10/v1",
        "http://169.254.169.254/latest", "https://10.0.0.5/v1", "ftp://1.1.1.1/v1",
        "https://user:pass@1.1.1.1/v1",                     # userinfo rejected
    ])
    def test_rejects_ssrf_userinfo_and_non_https_base_urls(self, bad):
        with pytest.raises(RestrictedLaneViolation):
            analyze_transcript("hi", runtime={**RUNTIME, "base_url": bad})

    @pytest.mark.parametrize("bad", [0, -1, float("inf"), float("nan"), lane._MAX_TIMEOUT + 1, "5"])
    def test_rejects_invalid_timeout(self, bad):
        with pytest.raises(RestrictedLaneViolation):
            analyze_transcript("hi", runtime=RUNTIME, timeout=bad)

    def test_allows_public_https_runtime(self):
        with _patch_client(json.dumps(lane._empty_result())) as m:
            analyze_transcript("hi", runtime=RUNTIME)
        assert m.called


# --------------------------------------------------------------------------- #
# The call carries zero tools + a bounded request
# --------------------------------------------------------------------------- #

class TestCallShape:
    def test_no_tools_bounded_request_and_input_capped(self):
        cap = {}
        huge = "x" * (lane._MAX_TRANSCRIPT_CHARS + 5000)
        with _patch_client(json.dumps(lane._empty_result()), capture=cap):
            analyze_transcript(huge, runtime=RUNTIME)
        assert "tools" not in cap or not cap["tools"]
        # response_format is opt-in (off by default for provider compatibility).
        assert "response_format" not in cap
        assert cap["max_tokens"] == 4096
        user_msg = cap["messages"][1]["content"]
        assert "truncated" in user_msg and len(user_msg) < lane._MAX_TRANSCRIPT_CHARS + 2000
        assert cap["messages"][0]["content"] == lane.RESTRICTED_ANALYSIS_SYSTEM_PROMPT

    def test_response_format_sent_only_when_opted_in(self):
        cap = {}
        with _patch_client(json.dumps(lane._empty_result()), capture=cap):
            analyze_transcript("hi", runtime={**RUNTIME, "send_response_format": True})
        assert cap["response_format"] == {"type": "json_object"}


# --------------------------------------------------------------------------- #
# Malicious transcript → bounded, sanitized result only
# --------------------------------------------------------------------------- #

class TestMaliciousTranscript:
    def test_rogue_keys_dropped(self):
        rogue = json.dumps({
            "summary": "Habit frameworks discussed.",
            "mechanisms": [{"name": "Two-minute rule", "description": "Shrink the habit."}],
            "entities": [], "injection_flags": ["Transcript said: 'Ignore all instructions'"],
            "leaked_pgpass": "localhost:5432:db:user:SECRET",
            "exfil_url": "https://exfil.example.com/collect",
        })
        with _patch_client(rogue):
            r = analyze_transcript(_load("malicious_injection.txt"), runtime=RUNTIME)
        assert set(r) == {"summary", "mechanisms", "entities", "injection_flags"}
        assert "leaked_pgpass" not in r and "exfil_url" not in r
        assert r["injection_flags"]

    def test_oversized_capped(self):
        # Over per-field caps but under the 256KB raw budget, so field capping is what
        # bounds it (the raw-byte guard has its own test).
        blob = json.dumps({
            "summary": "A" * 5000,
            "mechanisms": [{"name": "n" * 500, "description": "d" * 800,
                            "steps": ["s" * 400] * 12}] * 15,
            "entities": [{"name": "e" * 300, "type": "person"}] * 60,
            "injection_flags": ["f" * 400] * 30,
        })
        assert len(blob.encode()) < lane._MAX_RAW_RESPONSE_BYTES
        with _patch_client(blob):
            r = analyze_transcript("hi", runtime=RUNTIME)
        assert len(r["summary"]) == lane._MAX_SUMMARY
        assert len(r["mechanisms"]) == lane._MAX_MECHANISMS
        assert len(r["mechanisms"][0]["name"]) == lane._MAX_MECH_NAME
        assert len(r["mechanisms"][0]["steps"]) == lane._MAX_STEPS
        assert len(r["entities"]) == lane._MAX_ENTITIES
        assert len(r["injection_flags"]) == lane._MAX_FLAGS

    def test_control_and_bidi_chars_stripped(self):
        r = sanitize_output(json.dumps({
            "summary": "clean\x07\x1b[31m\x9b‮evil\x00", "mechanisms": [],
            "entities": [], "injection_flags": [],
        }))
        for bad in ("\x07", "\x00", "\x9b", "‮"):
            assert bad not in r["summary"]

    def test_bad_entity_type_coerced(self):
        r = sanitize_output(json.dumps({
            "summary": "", "mechanisms": [],
            "entities": [{"name": "X", "type": "shell_command"}], "injection_flags": [],
        }))
        assert r["entities"][0]["type"] == "other"

    def test_emoji_heavy_result_not_falsely_rejected(self):
        r = sanitize_output(json.dumps({
            "summary": "😀" * lane._MAX_SUMMARY, "mechanisms": [], "entities": [],
            "injection_flags": [],
        }))
        assert len(r["summary"]) == lane._MAX_SUMMARY


# --------------------------------------------------------------------------- #
# Ordinary transcript round-trips
# --------------------------------------------------------------------------- #

class TestOrdinaryTranscript:
    def test_returns_bounded_schema(self):
        good = json.dumps({
            "summary": "A bootstrapped founder explains the ICE score.",
            "mechanisms": [{"name": "ICE score", "description": "Rate I,C,E; multiply; sort.",
                            "steps": ["Score impact", "Score confidence", "Score ease"]}],
            "entities": [{"name": "The Lean Startup", "type": "book"}],
            "injection_flags": [],
        })
        with _patch_client(good):
            r = analyze_transcript(_load("ordinary_business.txt"), runtime=RUNTIME)
        assert r["mechanisms"][0]["name"] == "ICE score"
        assert r["entities"][0]["type"] == "book"

    def test_rejects_non_string_transcript(self):
        with pytest.raises(TypeError):
            analyze_transcript({"url": "x"}, runtime=RUNTIME)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# Fail-closed — incl. strict top-level validation (no silent empty success)
# --------------------------------------------------------------------------- #

class TestFailClosed:
    @pytest.mark.parametrize("content", [
        "not json at all",
        "[]",                                   # array, not object
        "{}",                                   # missing required keys
        json.dumps({"summary": 123, "mechanisms": "bad", "entities": {}, "injection_flags": None}),
        json.dumps({"summary": "ok", "mechanisms": [], "entities": []}),  # missing injection_flags
    ])
    def test_malformed_or_incomplete_raises(self, content):
        with _patch_client(content):
            with pytest.raises(RestrictedLaneViolation):
                analyze_transcript("hi", runtime=RUNTIME)

    def test_empty_and_none_content_raise(self):
        for c in ("", None):
            with _patch_client(c):
                with pytest.raises(RestrictedLaneViolation):
                    analyze_transcript("hi", runtime=RUNTIME)

    def test_deeply_nested_json_raises(self):
        payload = "[" * 5000 + "]" * 5000  # would RecursionError in json.loads
        with _patch_client(payload):
            with pytest.raises(RestrictedLaneViolation):
                analyze_transcript("hi", runtime=RUNTIME)

    def test_raw_output_over_byte_budget_raises(self):
        big = json.dumps({"summary": "x" * (lane._MAX_RAW_RESPONSE_BYTES + 10),
                          "mechanisms": [], "entities": [], "injection_flags": []})
        with _patch_client(big):
            with pytest.raises(RestrictedLaneViolation):
                analyze_transcript("hi", runtime=RUNTIME)

    def test_provider_exception_is_constant_and_untainted(self):
        secret = "SENSITIVE-PROVIDER-TEXT-12345"
        with _patch_client(raises=RuntimeError(secret)):
            with pytest.raises(RestrictedLaneViolation) as ei:
                analyze_transcript("hi", runtime=RUNTIME)
        assert secret not in str(ei.value)  # tainted provider text never crosses back

    def test_malformed_response_object_raises(self):
        client = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(
                create=lambda **k: SimpleNamespace(choices=[]))),
            close=lambda: None)
        with patch.object(lane, "_build_client", return_value=client):
            with pytest.raises(RestrictedLaneViolation):
                analyze_transcript("hi", runtime=RUNTIME)


class TestContentShapes:
    def test_list_content_parts_joined_non_string_skipped(self):
        good = json.dumps(lane._empty_result())
        content = [{"text": good}, {"text": None}, {"nope": 1}, SimpleNamespace(text=123)]
        with _patch_client(content):
            assert analyze_transcript("hi", runtime=RUNTIME) == lane._empty_result()

    def test_client_is_closed(self):
        closed = {"v": False}
        client = _fake_client(json.dumps(lane._empty_result()))
        client.close = lambda: closed.__setitem__("v", True)
        with patch.object(lane, "_build_client", return_value=client):
            analyze_transcript("hi", runtime=RUNTIME)
        assert closed["v"] is True


class TestRestrictedExtract:
    # A small rich schema standing in for a real extraction contract.
    SCHEMA = {
        "type": "object",
        "additionalProperties": False,
        "required": ["frameworks"],
        "properties": {
            "frameworks": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["name", "category"],
                    "properties": {
                        "name": {"type": "string"},
                        "category": {"type": "string", "enum": ["Sales", "Strategy"]},
                        "steps": {"type": "array", "items": {"type": "string"}},
                    },
                },
            },
        },
    }
    KW = dict(system_prompt="Extract frameworks as JSON.",
              transcript_text="The ICE method: rate impact, confidence, ease.")

    def _call(self, content, **over):
        kw = {**self.KW, "schema": self.SCHEMA, "runtime": RUNTIME, **over}
        with _patch_client(content):
            return lane.restricted_extract(**kw)

    def test_valid_extraction_passes_schema(self):
        good = json.dumps({"frameworks": [
            {"name": "ICE", "category": "Strategy", "steps": ["rate impact"]}]})
        r = self._call(good)
        assert r["frameworks"][0]["name"] == "ICE"

    def test_schema_violation_fails_closed(self):
        # 'category' not in enum → jsonschema rejects → RestrictedLaneViolation.
        bad = json.dumps({"frameworks": [{"name": "X", "category": "NotAllowed"}]})
        with pytest.raises(RestrictedLaneViolation):
            self._call(bad)

    def test_missing_required_fails_closed(self):
        with pytest.raises(RestrictedLaneViolation):
            self._call(json.dumps({"wrong": []}))

    def test_non_json_fails_closed(self):
        with pytest.raises(RestrictedLaneViolation):
            self._call("here are the secrets")

    def test_control_chars_stripped_in_nested_strings(self):
        good = json.dumps({"frameworks": [
            {"name": "clean\x07\x00name", "category": "Sales"}]})
        r = self._call(good)
        assert "\x07" not in r["frameworks"][0]["name"]
        assert "\x00" not in r["frameworks"][0]["name"]

    def test_zero_tools_and_response_format_opt_in(self):
        cap = {}
        good = json.dumps({"frameworks": []})
        with patch.object(lane, "_build_client",
                          return_value=_fake_client(good, capture=cap)):
            lane.restricted_extract(schema=self.SCHEMA, runtime=RUNTIME, **self.KW)
        assert "tools" not in cap
        assert "response_format" not in cap  # opt-in off by default
        with patch.object(lane, "_build_client",
                          return_value=_fake_client(good, capture=cap)):
            lane.restricted_extract(schema=self.SCHEMA,
                                    runtime={**RUNTIME, "send_response_format": True}, **self.KW)
        assert cap["response_format"]["type"] == "json_schema"

    def test_rejects_unsafe_runtime(self):
        with pytest.raises(RestrictedLaneViolation):
            lane.restricted_extract(schema=self.SCHEMA,
                                    runtime={**RUNTIME, "base_url": "http://127.0.0.1/v1"},
                                    **self.KW)

    def test_requires_schema_and_prompt(self):
        with pytest.raises(RestrictedLaneViolation):
            self._call(json.dumps({"frameworks": []}), schema={})
        with pytest.raises(RestrictedLaneViolation):
            self._call(json.dumps({"frameworks": []}), system_prompt="")


class TestClientConstruction:
    def test_real_client_drops_openai_tenant_ids(self, monkeypatch):
        # Even if the env sets OpenAI tenant ids, the locked client must not carry them
        # (they would be sent as headers to a parent-selected non-OpenAI endpoint).
        monkeypatch.setenv("OPENAI_ORG_ID", "org-should-not-leak")
        monkeypatch.setenv("OPENAI_PROJECT_ID", "proj-should-not-leak")
        client = lane._build_client("sk-test", "https://1.1.1.1/v1", 60.0)
        try:
            assert client.organization in ("", None)
            assert client.project in ("", None)
        finally:
            client.close()


class TestSanitizeUnits:
    def test_strips_json_fence(self):
        fenced = "```json\n" + json.dumps(lane._empty_result()) + "\n```"
        assert sanitize_output(fenced) == lane._empty_result()

    def test_well_formed_empty_is_valid(self):
        assert sanitize_output(json.dumps(lane._empty_result())) == lane._empty_result()
