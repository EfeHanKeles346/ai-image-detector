import hashlib

import pytest

from experiments import e46_acquisition as acquisition


def _publisher_csv() -> bytes:
    rows = ["url,filename,typ"]
    for typ, count in acquisition.SYNTHWILDX_COUNTS.items():
        for index in range(count):
            rows.append(f"https://pbs.twimg.com/media/{typ}{index}?format=jpg&name=large,{typ}/{index}.jpg,{typ}")
    return ("\n".join(rows) + "\n").encode()


def test_parse_synthwildx_list_and_roles(monkeypatch):
    raw = _publisher_csv()
    monkeypatch.setattr(acquisition, "SYNTHWILDX_LIST_BYTES", len(raw))
    monkeypatch.setattr(acquisition, "SYNTHWILDX_LIST_SHA256", hashlib.sha256(raw).hexdigest())
    parsed = acquisition.parse_synthwildx_list(raw)
    roles = acquisition.assign_roles(parsed)
    assert len(parsed) == 2000
    assert sum(value == "CAL" for value in roles.values()) == 1200
    assert sum(value == "DEVELOPMENT" for value in roles.values()) == 800
    for typ in acquisition.SYNTHWILDX_COUNTS:
        names = [row["filename"] for row in parsed if row["typ"] == typ]
        assert sum(roles[name] == "CAL" for name in names) == 300


def test_parse_synthwildx_fails_on_source_drift(monkeypatch):
    raw = _publisher_csv().replace(b"pbs.twimg.com", b"example.com", 1)
    monkeypatch.setattr(acquisition, "SYNTHWILDX_LIST_BYTES", len(raw))
    monkeypatch.setattr(acquisition, "SYNTHWILDX_LIST_SHA256", hashlib.sha256(raw).hexdigest())
    with pytest.raises(ValueError, match="host"):
        acquisition.parse_synthwildx_list(raw)


def test_parse_drive_landing():
    body = f'''<a href="/open?id={acquisition.TRUEFAKE_FILE_ID}">Facebook.tar.gz</a> (3.9G)
    <input name="id" value="{acquisition.TRUEFAKE_FILE_ID}">
    <input name="confirm" value="t"><input name="uuid" value="abc">'''
    fields = acquisition.parse_drive_landing(body)
    assert fields == {"id": acquisition.TRUEFAKE_FILE_ID, "confirm": "t", "uuid": "abc"}


@pytest.mark.parametrize("change", ["id", "confirm", "filename", "size"])
def test_parse_drive_landing_fails_closed(change):
    values = {
        "id": acquisition.TRUEFAKE_FILE_ID,
        "confirm": "t",
        "uuid": "abc",
        "filename": "Facebook.tar.gz",
        "size": "3.9G",
    }
    values[change] = "changed"
    body = f'''<a href="/open?id={values['id']}">{values['filename']}</a> ({values['size']})
    <input name="id" value="{values['id']}"><input name="confirm" value="{values['confirm']}">
    <input name="uuid" value="{values['uuid']}">'''
    with pytest.raises(ValueError):
        acquisition.parse_drive_landing(body)


def test_parse_drive_landing_requires_uuid():
    body = f'''<a href="/open?id={acquisition.TRUEFAKE_FILE_ID}">Facebook.tar.gz</a> (3.9G)
    <input name="id" value="{acquisition.TRUEFAKE_FILE_ID}">
    <input name="confirm" value="t"><input name="uuid" value="">'''
    with pytest.raises(ValueError):
        acquisition.parse_drive_landing(body)
