import numpy as np
import pytest

from experiments.e60_run import pair_rates, disable_network


def test_real_improvement_does_not_hide_new_ai_errors():
    result = pair_rates(np.array([0, 0, 1, 1]), np.array(['camera', 'camera', 'ai', 'ai']),
                        np.array([.2, .001, .9, .2]), np.array([.001, .001, .9, .001]))
    assert result['real_rescued'] == 1
    assert result['ai_new_errors'] == 1
    assert result['new']['real_false_ai'] < result['old']['real_false_ai']
    assert result['new']['ai_recall'] < result['old']['ai_recall']
    assert result['by_source']['1:ai']['new_ai_minus_old'] == -.5


def test_network_guard_blocks_connect_without_attempting_external_access(monkeypatch):
    import socket
    for attr in ('connect', 'connect_ex'):
        monkeypatch.setattr(socket.socket, attr, getattr(socket.socket, attr))
    monkeypatch.setattr(socket, 'create_connection', socket.create_connection)
    for env in ('HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE', 'HF_HUB_DISABLE_TELEMETRY'):
        monkeypatch.setenv(env, '0')
    disable_network()
    with pytest.raises(RuntimeError, match='disabled'):
        socket.create_connection(('example.invalid', 443))
