from pathlib import Path

from orchid_ranker.security import AccessControl, AuditLogger


def test_access_control_enforces_permissions(tmp_path):
    acl = AccessControl()
    assert acl.can("ml_engineer", "preprocess")
    assert not acl.can("viewer", "experiment")
    try:
        acl.require("viewer", "experiment")
    except PermissionError as exc:
        assert "viewer" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("viewer should not be allowed to run experiments")


def test_audit_logger_writes_json(tmp_path):
    log_path = tmp_path / "audit.jsonl"
    logger = AuditLogger(Path(log_path))
    logger.log_event("dp_update", actor="tester", payload={"epsilon_cum": 0.5})

    contents = log_path.read_text().strip().splitlines()
    assert len(contents) == 1
    assert "dp_update" in contents[0]
    assert "tester" in contents[0]


def test_audit_logger_remote_hook(tmp_path):
    sent = []

    class DummyAuditLogger(AuditLogger):
        def _send_remote(self, event_json: str) -> None:  # type: ignore[override]
            sent.append(event_json)

    log_path = tmp_path / "remote.jsonl"
    logger = DummyAuditLogger(Path(log_path), endpoint="https://example.com")
    logger.log_event("dp_update", actor="tester", payload={"epsilon_delta": 0.1})

    assert sent, "expected remote payload to be captured"
    assert "epsilon_delta" in sent[0]


def test_audit_logger_from_env(tmp_path, monkeypatch):
    monkeypatch.setenv("ORCHID_AUDIT_ENDPOINT", "https://example.com/webhook")
    monkeypatch.setenv("ORCHID_AUDIT_API_KEY", "secret")
    monkeypatch.setenv("ORCHID_AUDIT_TIMEOUT", "5.5")

    logger = AuditLogger.from_env(Path(tmp_path / "env.jsonl"))
    assert logger.endpoint == "https://example.com/webhook"
    assert logger.api_key == "secret"
    assert logger.timeout == 5.5

    calls = []

    def fake_send(event_json: str) -> None:
        calls.append(event_json)

    monkeypatch.setattr(logger, "_send_remote", fake_send)
    logger.log_event("dp_update", actor="env", payload={})
    assert calls, "expected event to be forwarded"
