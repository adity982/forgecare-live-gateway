from pathlib import Path

import numpy as np

from forgecare import phone_federation_schedule as schedule


def test_two_clients_keep_physical_samples_separate():
    labels = np.asarray([0] * 12 + [1] * 12)
    groups = np.asarray([f"good-{i // 2}" for i in range(12)] + [f"bad-{i // 2}" for i in range(12)])
    parts = schedule.two_clients(labels, groups, np.arange(24))
    assert all(set(labels[part]) == {0, 1} for part in parts)
    assert not set(groups[parts[0]]) & set(groups[parts[1]])


def test_scheduled_tick_skips_unusable_same_phone_labels(monkeypatch, tmp_path):
    monkeypatch.setattr(schedule, "select_rows", lambda site, asset: ([], {
        "reviewed_photos": 5, "usable_labeled_photos": 0,
        "distinct_acceptable_phones": 0, "distinct_damaged_phones": 0,
        "conflicting_phone_ids": 1}))
    result = schedule.train_asset("site", {"id": "iphone", "model": "iPhone 14 Plus"}, tmp_path)
    assert result["status"] == "skipped_insufficient_distinct_reviewed_phones"
    assert not list(tmp_path.glob("*.joblib"))


def test_fedavg_trains_and_saves_model_only_after_grouped_check(monkeypatch, tmp_path):
    records = []
    features = {}
    for label, prefix in ((0, "good"), (1, "bad")):
        for phone in range(9):
            for photo in range(3):
                path = tmp_path / f"{prefix}-{phone}-{photo}.jpg"
                path.write_bytes(f"{prefix}-{phone}-{photo}".encode())
                records.append({"id": path.stem, "image_path": str(path),
                                "sample_id": f"{prefix}-{phone}", "binary_label": label})
                vector = np.zeros(schedule.VISUAL_FEATURE_COUNT)
                vector[0] = .2 + .5 * label + .01 * photo
                features[str(path)] = vector
    monkeypatch.setattr(schedule, "select_rows", lambda site, asset: (records, {
        "reviewed_photos": len(records), "usable_labeled_photos": len(records),
        "distinct_acceptable_phones": 9, "distinct_damaged_phones": 9,
        "conflicting_phone_ids": 0}))
    monkeypatch.setattr(schedule, "image_features", lambda path: features[str(path)])
    result = schedule.train_asset("site", {"id": "iphone", "model": "iPhone 14 Plus"}, tmp_path)
    assert result["status"] == "completed_local_research_model"
    assert len(result["rounds"]) == 3
    assert result["train_photos"] + result["validation_photos"] + result["test_photos"] == len(records)
    assert Path(result["model_path"]).exists()
