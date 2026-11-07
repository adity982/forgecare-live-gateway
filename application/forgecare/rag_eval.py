"""Reviewed retrieval gold set evaluation; no synthetic correctness labels."""
from __future__ import annotations

import csv

from .config import DATA
from .knowledge_api import search


def evaluate(site_id: str) -> dict:
    path = DATA / "rag_gold.csv"
    if not path.exists():
        return {"status": "insufficient_data", "reason": "Create data/rag_gold.csv with reviewed question,family,model,relevant_passage_id rows",
                "metrics": None}
    with path.open(encoding="utf-8-sig", newline="") as source:
        records = list(csv.DictReader(source))
    required = {"question", "family", "model", "relevant_passage_id"}
    if len(records) < 5 or any(not required.issubset(item) for item in records):
        return {"status": "insufficient_data", "reason": "At least five reviewed, complete gold questions are required", "metrics": None}
    details = []
    for item in records:
        matches = search(site_id, item["family"], item["model"], item["question"], limit=5)
        expected = item["relevant_passage_id"]
        ids = [match["id"] for match in matches]
        details.append({"question": item["question"], "relevant_passage_id": expected,
                        "retrieved_ids": ids, "hit_at_5": expected in ids,
                        "precision_at_5": (1 / len(ids)) if expected in ids and ids else 0})
    return {"status": "complete", "dataset": "human_reviewed_site_gold_questions",
            "n": len(details), "recall_at_5": round(sum(x["hit_at_5"] for x in details) / len(details), 4),
            "precision_at_5": round(sum(x["precision_at_5"] for x in details) / len(details), 4),
            "details": details,
            "answer_faithfulness": "requires_human_source_review"}
