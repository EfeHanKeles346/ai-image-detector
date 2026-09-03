from experiments.e47_gan_recovery import analyze


def _row(record_id, label, source, score):
    return {"record_id": record_id, "label": label, "source": source, "score": score, "status": "ok"}


def test_analysis_reports_complementarity_and_unlocks_strong_gan_arm():
    rows = []
    fused = []
    for index in range(20):
        rows.append(_row(f"r{index}", 0, "camera", index / 100))
        fused.append(_row(f"r{index}", 0, "camera", 0.1))
    for source in ("StyleGAN", "StyleGAN2", "StyleGAN3"):
        for index in range(10):
            record_id = f"{source}-{index}"
            rows.append(_row(record_id, 1, source, 0.8 + index / 100))
            fused.append(_row(record_id, 1, source, 0.1))

    result = analyze(rows, fused)

    assert result["unlock"]["passed"] is True
    assert result["complementarity"]["misses_recovered"] == 30
    assert result["legacy"]["ai_recall_by_source"]["StyleGAN2"] == 1.0


def test_analysis_rejects_weak_gan_arm():
    rows = []
    fused = []
    for index in range(20):
        rows.append(_row(f"r{index}", 0, "camera", 0.4 + index / 100))
        fused.append(_row(f"r{index}", 0, "camera", 0.1))
    for source in ("StyleGAN", "StyleGAN2", "StyleGAN3"):
        for index in range(10):
            record_id = f"{source}-{index}"
            rows.append(_row(record_id, 1, source, 0.1 + index / 100))
            fused.append(_row(record_id, 1, source, 0.1))

    result = analyze(rows, fused)

    assert result["unlock"]["passed"] is False
    assert result["unlock"]["mean_stylegan_recall_gte_0_50"] is False
