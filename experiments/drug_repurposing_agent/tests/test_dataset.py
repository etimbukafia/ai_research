from __future__ import annotations

from experiments.drug_repurposing_agent.src.runner import validate_data


def test_frozen_dataset_validates() -> None:
    report = validate_data()

    assert report["pairs"] == 100
    assert report["gold_records"] == 100
    assert report["unique_pairs"] == 100
    assert report["split_overlap"] is False
    assert report["missing_gold_fields"] is False
    assert report["hashes_verified"] is True
    assert report["leakage_policy_verified"] is True
