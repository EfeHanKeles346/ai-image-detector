import json
import socket

import pytest

from experiments import e61_replay_gate as replay


def test_changed_bound_input_fails_before_model_loading(tmp_path, monkeypatch):
    bound = tmp_path / 'input'; bound.write_text('modified')
    contract = tmp_path / 'contract.json'
    contract.write_text(json.dumps({'inputs': {'runner': {'path': str(bound), 'sha256': '0' * 64}}}))
    monkeypatch.setattr(replay, 'CONTRACT', contract)
    monkeypatch.setattr(replay, 'RESULT', tmp_path / 'result.json')
    with pytest.raises(ValueError, match='changed bound input'):
        replay.run()
    assert not replay.RESULT.exists()


def test_evidence_cannot_be_overwritten(tmp_path):
    path = tmp_path / 'result.json'
    replay.write_once(path, {'state': 'original'})
    with pytest.raises(FileExistsError):
        replay.write_once(path, {'state': 'replacement'})
    assert json.loads(path.read_text())['state'] == 'original'


def test_network_guard_blocks_before_connect(monkeypatch):
    for attr in ('connect', 'connect_ex'):
        monkeypatch.setattr(socket.socket, attr, getattr(socket.socket, attr))
    monkeypatch.setattr(socket, 'create_connection', socket.create_connection)
    for env in ('HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE', 'HF_HUB_DISABLE_TELEMETRY'):
        monkeypatch.setenv(env, '0')
    replay.disable_network()
    with pytest.raises(RuntimeError, match='disabled'):
        socket.create_connection(('example.invalid', 443))
