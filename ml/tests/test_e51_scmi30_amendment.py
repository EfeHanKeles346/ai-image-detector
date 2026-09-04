from experiments.e51_scmi30_amendment import reserve_rows


def test_replacement_reserve_reuses_original_score_blind_rank():
    rows = [{"name": f"SCMI30-IITRPR/Similar/D04_Samsung_Galaxy_M04/natural/{i}.jpg",
             "total_bytes": i + 1} for i in range(10)]
    first = reserve_rows(rows, {rows[0]["name"]}, count=3)
    second = reserve_rows(list(reversed(rows)), {rows[0]["name"]}, count=3)
    assert first == second
    assert len(first) == 3
    assert all(row["device_id"] == "D04" and row["branch"] == "Similar" for row in first)
