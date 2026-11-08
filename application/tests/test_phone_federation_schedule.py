import json
import sys

from forgecare.federation import simulate_synthetic_federation
from forgecare import federated_client
from forgecare import phone_federation_schedule as schedule


def test_synthetic_federation_uses_five_rounds_and_keeps_rows_local():
    result = simulate_synthetic_federation()

    assert result["dataset"] == "newly_generated_synthetic_examples_in_memory"
    assert result["raw_data_shared"] is False
    assert result["round_count"] == 5
    assert [item["round"] for item in result["rounds"]] == [1, 2, 3, 4, 5]
    assert set(result["clients"]) == {"Client A", "Client B"}
    assert all(item["client_updates"]["Client A"]["training_examples"] > 0
               for item in result["rounds"])


def test_scheduled_tick_writes_synthetic_report_to_requested_directory(tmp_path):
    report_path = schedule.run_tick(tmp_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))

    assert report_path.parent == tmp_path
    assert report["data_source"] == "newly generated synthetic examples in memory; no existing data read"
    assert report["simulation"]["round_count"] == 5
    assert len(report["simulation"]["rounds"]) == 5


def test_synthetic_client_cli_runs_without_manifest_or_credentials(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["federated_client", "synthetic-demo", "--pause-seconds", "0"])

    federated_client.run()

    output = capsys.readouterr().out
    result = json.loads(output[output.index("{"):])
    assert result["round_count"] == 5
    assert result["raw_data_shared"] is False
