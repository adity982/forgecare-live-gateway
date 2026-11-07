from fastapi.testclient import TestClient

from forgecare.config import SERVICE_KEY
from forgecare import knowledge_api
from forgecare import agent
from forgecare.groq_provider import GroqUnavailable
import numpy as np


def test_site_model_revision_filter_and_abstention(monkeypatch):
    html = b"<html><head><title>iPhone exterior guide</title></head><body><main><h1>Screen inspection</h1><p>For visible cracked glass, stop handling the device and ask a trained technician to review the applicable repair guidance.</p><p>Inspect the exterior in a well-lit area and record the observed damage before any maintenance decision.</p></main></body></html>"

    class Response:
        text = html.decode()
        content = html
        def raise_for_status(self): pass

    monkeypatch.setattr(knowledge_api.requests, "get", lambda *args, **kwargs: Response())
    monkeypatch.setattr(knowledge_api, "embedding_model", lambda: (_ for _ in ()).throw(RuntimeError("offline")))
    monkeypatch.setattr(knowledge_api, "local_generate", lambda q, evidence: (None, "model unavailable in test"))
    headers = {"X-Service-Key": SERVICE_KEY}
    with TestClient(knowledge_api.app) as client:
        added = client.post("/documents", headers=headers, json={"site_id": "a", "family": "iphone-exterior", "model": "iPhone 17", "url": "https://support.apple.com/guide/iphone/test", "revision": "r1"})
        assert added.status_code == 200, added.text
        realme = client.post("/documents", headers=headers, json={"site_id": "a", "family": "realme-exterior", "model": "realme narzo 60 5G", "url": "https://www.realme.com/in/realme-narzo-60/specs", "revision": "r1"})
        assert realme.status_code == 200, realme.text
        assert client.post("/documents", headers=headers, json={"site_id": "a", "family": "iphone-exterior", "url": "https://evil.example/test"}).status_code == 422
        found = client.post("/ask", headers=headers, json={"site_id": "a", "family": "iphone-exterior", "model": "iPhone 17", "question": "What about cracked glass?"}).json()
        assert found["status"] == "retrieval_only" and found["evidence"]
        for wrong in ({"site_id": "b", "model": "iPhone 17"}, {"site_id": "a", "model": "iPhone 16"}):
            result = client.post("/ask", headers=headers, json={"family": "iphone-exterior", "question": "cracked glass", **wrong}).json()
            assert result["status"] == "abstained"
        assert client.post("/ask", headers=headers, json={"site_id": "a", "family": "realme-exterior", "model": "iPhone 17", "question": "cracked glass"}).json()["status"] == "abstained"
        client.post(f"/documents/{added.json()['id']}/deactivate", headers=headers)
        result = client.post("/ask", headers=headers, json={"site_id": "a", "family": "iphone-exterior", "model": "iPhone 17", "question": "cracked glass"}).json()
        assert result["status"] == "abstained"


def test_agent_does_not_delegate_tool_choice_to_model(monkeypatch):
    monkeypatch.setattr(agent, "generate_text", lambda *args, **kwargs: '{"tool":"run_shell"}')
    monkeypatch.setattr(agent, "search", lambda *args, **kwargs: [])
    result = agent.prepare({"site_id": "a", "asset": {"id": "asset", "family": "iphone-exterior"}}, "Prepare inspection")
    assert result["status"] == "abstained_no_applicable_document"
    assert result["draft"] is None
    assert [item["tool"] for item in result["trace"]] == ["get_asset", "get_history", "get_queue", "search_manuals"]


def test_agent_unanswered_task_lists_applicable_references(monkeypatch):
    monkeypatch.setattr(agent, "search", lambda *args, **kwargs: [])
    with knowledge_api.open_db() as db:
        db.execute("INSERT INTO documents VALUES(?,?,?,?,?,?,?,?,?,?)",
                   ("manual", "a", "iphone-exterior", "iPhone 14 Plus", "Apple display guide",
                    "https://support.apple.com/en-us/100329", "r1", "active", "now", "hash"))
    result = agent.prepare({"site_id": "a", "asset": {"id": "iphone", "family": "iphone-exterior",
                                                    "model": "iPhone 14 Plus"}}, "Unanswered task request")
    assert result["status"] == "abstained_no_applicable_document"
    assert [source["title"] for source in result["searched_sources"]] == ["Apple display guide"]


def test_agent_produces_review_only_extractive_draft(monkeypatch):
    monkeypatch.setattr(agent, "configured", lambda: False)
    monkeypatch.setattr(agent, "generate_text", lambda *args, **kwargs: "Uncited generated instruction")
    passage_id = "11111111-1111-1111-1111-111111111111"
    monkeypatch.setattr(agent, "search", lambda *args, **kwargs: [{"id": passage_id, "text": "Inspect visible glass in good light.", "url": "https://support.apple.com/test"}])
    result = agent.prepare({"site_id": "a", "asset": {"id": "asset", "family": "iphone-exterior"}, "case": None,
                            "history": [], "queue": []}, "Prepare inspection")
    assert result["status"] == "pending_approval_extractive"
    assert passage_id in result["draft"]
    assert "Uncited generated instruction" not in result["draft"]
    assert [item["tool"] for item in result["trace"]] == ["get_asset", "get_history", "get_queue", "search_manuals", "save_draft"]


def test_agent_uses_source_packet_when_generation_fails(monkeypatch):
    monkeypatch.setattr(agent, "configured", lambda: True)
    monkeypatch.setattr(agent, "complete", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("provider unavailable")))
    passage_id = "22222222-2222-2222-2222-222222222222"
    monkeypatch.setattr(agent, "search", lambda *args, **kwargs: [{"id": passage_id, "text": "Record visible glass condition.", "url": "https://support.apple.com/test"}])
    result = agent.prepare({"site_id": "a", "asset": {"id": "asset", "family": "iphone-exterior"}}, "Record visible glass condition")
    assert result["status"] == "pending_approval_extractive"
    assert passage_id in result["draft"]
    assert [item["tool"] for item in result["trace"]][-2:] == ["search_manuals", "save_draft"]


def test_dense_results_preserve_passage_ids(monkeypatch):
    with knowledge_api.open_db() as db:
        db.execute("INSERT INTO documents VALUES(?,?,?,?,?,?,?,?,?,?)",
                   ("d1", "a", "iphone-exterior", "*", "Guide", "https://support.apple.com/test", "r1", "active", "now", "hash"))
        db.execute("INSERT INTO passages VALUES(?,?,?,?)", ("irrelevant", "d1", 0, "Battery charging information and warranty notice."))
        db.execute("INSERT INTO passages VALUES(?,?,?,?)", ("relevant", "d1", 1, "Relevant visual inspection procedure for an exterior component."))

    class Encoder:
        def encode(self, texts, **kwargs):
            return np.array([[1, 0] if text == "unseenquery" or "Relevant visual" in text else [0, 1] for text in texts], dtype="float32")

    monkeypatch.setattr(knowledge_api, "embedding_model", lambda: Encoder())
    results = knowledge_api.search("a", "iphone-exterior", "iPhone 17", "unseenquery")
    assert results[0]["id"] == "relevant"
    assert results[0]["retrieval"] == "hybrid_faiss"
def test_groq_answer_requires_approved_source_citation(monkeypatch):
    from forgecare import knowledge_api
    passage = {"id": "12345678-1234-1234-1234-123456789abc", "title": "Approved guide",
               "url": "https://example.com/guide", "text": "Inspect the visible screen under good light."}
    monkeypatch.setattr(knowledge_api, "configured", lambda: True)
    monkeypatch.setattr(knowledge_api, "complete", lambda *args, **kwargs: "Inspect under good light. [12345678-1234-1234-1234-123456789abc]")
    answer, provider = knowledge_api.local_generate("How should I inspect it?", [passage])
    assert provider == "generated_with_groq" and passage["id"] in answer
    monkeypatch.setattr(knowledge_api, "complete", lambda *args, **kwargs: "Inspect under good light.")
    answer, reason = knowledge_api.local_generate("How should I inspect it?", [passage])
    assert answer is None and "citations" in reason
    monkeypatch.setattr(knowledge_api, "complete", lambda *args, **kwargs: "The approved sources do not establish that answer.")
    answer, reason = knowledge_api.local_generate("What is the battery screw torque?", [passage])
    assert answer is None and reason == "unsupported_by_approved_sources"


def test_groq_rate_limit_uses_second_model(monkeypatch):
    requested = []
    def answer(messages, *, model, max_tokens):
        requested.append(model)
        if model == knowledge_api.TEXT_MODEL:
            raise GroqUnavailable("Groq returned HTTP 429")
        return "Source-backed answer."
    monkeypatch.setattr(knowledge_api, "complete", answer)
    assert knowledge_api.complete_with_fallback([], 100) == "Source-backed answer."
    assert requested == [knowledge_api.TEXT_MODEL, knowledge_api.TEXT_FALLBACK_MODEL]


def test_unanswered_question_lists_only_applicable_approved_references(monkeypatch):
    monkeypatch.setattr(knowledge_api, "search", lambda *args, **kwargs: [])
    with knowledge_api.open_db() as db:
        for doc_id, site_id, family in (("relevant", "a", "manufacturing-general"),
                                        ("other-site", "b", "manufacturing-general"),
                                        ("other-topic", "a", "iphone-general")):
            db.execute("INSERT INTO documents VALUES(?,?,?,?,?,?,?,?,?,?)",
                       (doc_id, site_id, family, "*", doc_id, "https://www.nist.gov/" + doc_id,
                        "r1", "active", "now", "hash"))
    result = knowledge_api.ask(knowledge_api.AskIn(
        site_id="a", family="iphone-exterior", model="iPhone 14 Plus",
        question="How is this factory process checked?"))
    assert result["status"] == "abstained" and result["answer"] is None
    assert [source["title"] for source in result["searched_sources"]] == ["relevant"]


def test_general_questions_route_to_their_own_approved_sources(monkeypatch):
    monkeypatch.setattr(knowledge_api, "embedding_model", lambda: (_ for _ in ()).throw(RuntimeError("offline")))
    monkeypatch.setattr(knowledge_api, "local_generate", lambda q, evidence: (None, "generation unavailable"))
    sources = [
        ("ios", "iphone-general", "Battery health on iPhone", "iPhone battery health shows maximum capacity."),
        ("android", "android-general", "Android charging", "Check the Android charging cable and power adapter."),
        ("manufacturing", "manufacturing-general", "Process control charts", "A control chart monitors process variation over time."),
        ("automotive", "automotive-general", "Automotive FMEA", "FMEA identifies automotive product and process failure modes."),
    ]
    with knowledge_api.open_db() as db:
        for doc_id, family, title, passage in sources:
            db.execute("INSERT INTO documents VALUES(?,?,?,?,?,?,?,?,?,?)",
                       (doc_id, "a", family, "*", title, "https://example.com/" + doc_id,
                        "r1", "active", "now", "hash"))
            db.execute("INSERT INTO passages VALUES(?,?,?,?)", (doc_id + "-passage", doc_id, 0, passage))
    headers = {"X-Service-Key": SERVICE_KEY}
    with TestClient(knowledge_api.app) as client:
        base = {"site_id": "a", "family": "iphone-exterior", "model": "iPhone 14 Plus"}
        phone = client.post("/ask", headers=headers, json={**base, "question": "What does iPhone battery health mean?"}).json()
        assert phone["scope"] == "phone" and [x["document_id"] for x in phone["evidence"]] == ["ios"]
        factory = client.post("/ask", headers=headers, json={**base, "question": "What is a process control chart?"}).json()
        assert factory["scope"] == "manufacturing"
        assert [x["document_id"] for x in factory["evidence"]] == ["manufacturing"]
        auto = client.post("/ask", headers=headers, json={**base, "question": "What is FMEA in automotive quality?"}).json()
        assert auto["scope"] == "automotive"
        assert [x["document_id"] for x in auto["evidence"]] == ["automotive"]
        android = client.post("/ask", headers=headers, json={"site_id": "a", "family": "realme-exterior",
            "model": "realme narzo 60 5G", "question": "How can I check Android charging?", "scope": "phone"}).json()
        assert [x["document_id"] for x in android["evidence"]] == ["android"]
        exact = client.post("/ask", headers=headers, json={**base,
            "question": "What is the exact torque for an unknown car engine bolt?"}).json()
        assert exact["status"] == "abstained" and "component" in exact["reason"]


def test_generated_answer_retries_when_claims_are_only_cited_at_the_end(monkeypatch):
    passage_id = "12345678-1234-1234-1234-123456789abc"
    evidence = [{"id": passage_id, "title": "Approved guide", "url": "https://example.com/guide",
                 "text": "A control chart monitors a quality characteristic across time."}]
    answers = iter([f"A control chart monitors quality. It can show trends. [{passage_id}]",
                    f"A control chart monitors a quality characteristic over time. [{passage_id}]"])
    monkeypatch.setattr(knowledge_api, "configured", lambda: True)
    monkeypatch.setattr(knowledge_api, "complete", lambda *args, **kwargs: next(answers))
    answer, status = knowledge_api.local_generate("What is a control chart?", evidence)
    assert status == "generated_with_groq"
    assert answer == f"A control chart monitors a quality characteristic over time. [{passage_id}]"


def test_repair_question_retrieves_exact_model_procedure_and_prerequisites(monkeypatch):
    monkeypatch.setattr(knowledge_api, "embedding_model", lambda: (_ for _ in ()).throw(RuntimeError("offline")))
    with knowledge_api.open_db() as db:
        db.execute("INSERT INTO documents VALUES(?,?,?,?,?,?,?,?,?,?)",
                   ("health", "a", "iphone-general", "*", "iPhone battery and performance",
                    "https://support.apple.com/health", "r1", "active", "now", "hash"))
        db.execute("INSERT INTO passages VALUES(?,?,?,?)",
                   ("health-text", "health", 0, "iPhone battery performance health and capacity information."))
        db.execute("INSERT INTO documents VALUES(?,?,?,?,?,?,?,?,?,?)",
                   ("procedure", "a", "iphone-exterior", "iPhone 14 Plus", "iPhone 14 Plus Battery",
                    "https://support.apple.com/procedure", "r1", "active", "now", "hash"))
        db.execute("INSERT INTO passages VALUES(?,?,?,?)",
                   ("prerequisites", "procedure", 0, "Battery replacement requires trained technicians. Read battery safety and remove the back glass before beginning."))
        db.execute("INSERT INTO passages VALUES(?,?,?,?)",
                   ("steps", "procedure", 1, "Battery removal and reassembly use the approved repair procedure and battery press."))
    found = knowledge_api.search("a", "iphone-exterior", "iPhone 14 Plus",
                                  "How to change an iphone battery", extra_families=("iphone-general",))
    assert found[0]["document_id"] == "procedure"
    assert found[0]["ordinal"] == 0
    assert "back glass" in knowledge_api.expanded_query("How to change the back screen")


def test_explicit_iphone_question_uses_iphone_guidance_even_if_other_asset_selected(monkeypatch):
    from forgecare import ops_api
    sent = []

    class Response:
        def raise_for_status(self):
            pass
        def json(self):
            return {"status": "retrieval_only", "answer": None, "evidence": []}

    def fake_post(url, **kwargs):
        sent.append(kwargs["json"])
        return Response()

    monkeypatch.setattr(ops_api.requests, "post", fake_post)
    with TestClient(ops_api.app) as client:
        token = client.post("/setup/bootstrap", json={"site_id": "site-a", "site_name": "Demo",
            "username": "admin", "password": "super-secure-password"}).json()["token"]
        headers = {"Authorization": f"Bearer {token}"}
        for asset_id, family, model in (("iphone", "iphone-exterior", "iPhone 14 Plus"),
                                         ("realme", "realme-exterior", "realme narzo 60 5G")):
            client.post("/assets", headers=headers, json={"id": asset_id, "site_id": "site-a",
                "name": model, "family": family, "model": model, "criticality": 3})
        answer = client.post("/assistant/ask", headers=headers,
            json={"site_id": "site-a", "asset_id": "realme",
                  "question": "How to change an iPhone battery?"}).json()
    assert sent[0]["family"] == "iphone-exterior"
    assert sent[0]["model"] == "iPhone 14 Plus"
    assert answer["asset_model_used"] == "iPhone 14 Plus"


def test_official_iphone_repair_questions_have_cited_overviews():
    for question, title, text in (
        ("How to change an iphone battery", "iPhone 14 Plus Battery - Apple Support",
         "A battery should be replaced only by technicians with the knowledge and experience to repair electronic devices. Read Battery Safety before you begin."),
        ("How to change the back screen if it is damaged", "iPhone 14 Plus Back Glass - Apple Support",
         "Read Battery Safety and follow workspace guidelines before you begin. Read Broken Glass Safety before you begin."),
    ):
        passage = {"id": "12345678-1234-1234-1234-123456789abc", "ordinal": 0,
                   "title": title, "url": "https://support.apple.com/en-us/100346", "text": text}
        answer = knowledge_api.official_repair_overview(question, [passage])
        assert answer and knowledge_api.cites_each_claim(answer, {passage["id"]})
        assert "linked Apple" in answer


def test_iphone_battery_and_front_display_guidance_uses_exact_active_sources(monkeypatch):
    monkeypatch.setattr(knowledge_api, "embedding_model", lambda: (_ for _ in ()).throw(RuntimeError("offline")))
    documents = [
        ("health", "iphone-general", "*", "https://support.apple.com/en-us/101575",
         12, "To see Battery Health: Open the Settings app, then tap Battery. Review service guidance."),
        ("safety", "iphone-exterior", "iPhone 14 Plus", "https://support.apple.com/en-us/100299",
         1, "Don't use a battery with any visible damage, including dents, swelling, punctures, or deformation."),
        ("display", "iphone-exterior", "iPhone 14 Plus", "https://support.apple.com/en-us/100329",
         0, "Read Battery Safety and Broken Glass Safety before you begin. Dedicated display tools, removal and reassembly are required."),
        ("glass", "iphone-exterior", "iPhone 14 Plus", "https://support.apple.com/en-us/100298",
         0, "iPhone displays and back glass could break after impact. Don't remove chipped or cracked glass shards without protection."),
    ]
    with knowledge_api.open_db() as db:
        for index, (doc_id, family, model, url, ordinal, passage) in enumerate(documents, 1):
            db.execute("INSERT INTO documents VALUES(?,?,?,?,?,?,?,?,?,?)",
                       (doc_id, "a", family, model, f"iPhone 14 Plus {doc_id}", url, "r1", "active", "now", "hash"))
            db.execute("INSERT INTO passages VALUES(?,?,?,?)", (f"{index:08d}-1111-1111-1111-111111111111", doc_id, ordinal, passage))
    for question, expected in (
        ("How to check if the battery is damaged in the phone?", {"health", "safety"}),
        ("How to change the front glass of the phone?", {"display"}),
        ("How do I check for cracked glass on an iPhone 14 Plus?", {"glass"}),
    ):
        answer, evidence = knowledge_api.official_iphone_guidance("a", "iphone-exterior", "iPhone 14 Plus", question)
        assert {item["document_id"] for item in evidence} == expected
        assert knowledge_api.cites_each_claim(answer, {item["id"] for item in evidence})
        result = knowledge_api.ask(knowledge_api.AskIn(site_id="a", family="iphone-exterior",
                                                       model="iPhone 14 Plus", question=question))
        assert result["status"] == "generated" and result["provider"] == "approved sources"
        assert result["answer"] == answer
        assert knowledge_api.official_iphone_guidance("a", "realme-exterior", "realme narzo 60 5G", question) is None
    swollen, _ = knowledge_api.official_iphone_guidance(
        "a", "iphone-exterior", "iPhone 14 Plus", "What if the battery is swollen?")
    assert swollen.splitlines()[0].startswith("If a technician can see battery swelling")
