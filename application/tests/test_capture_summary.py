from forgecare.capture_summary import summarize
from forgecare import groq_provider
from PIL import Image


def test_multiview_draft_uses_largest_single_view_and_leaves_dents_unestablished():
    photos = {
        "front": {"counts": {"broken area": 1, "dent": 0, "major scratch": 1, "small scratch": 1},
                  "findings": [{"confidence": .7}], "runtime_ms": 31.2},
        "back": {"counts": {"broken area": 0, "dent": 0, "major scratch": 0, "small scratch": 1},
                 "findings": [{"confidence": .6}], "runtime_ms": 28.1},
    }
    result = summarize(photos, {"front": "front", "back": "back"})
    assert result["scratch_hint"] == 2
    assert result["dent_hint"] is None
    assert len(result["rows"]) == 2
    assert "Repeated views may show the same mark" in result["draft_text"]
    assert "Dent count could not be established" in result["draft_text"]


def test_zero_box_draft_does_not_claim_clean_phone():
    result = summarize({"photo": {"counts": {"broken area": 0, "dent": 0,
                        "major scratch": 0, "small scratch": 0}, "findings": [], "runtime_ms": 12}}, {})
    assert result["scratch_hint"] == 0
    assert "not a count of unique marks" in result["draft_text"]


def test_structured_vision_draft_keeps_unsupported_count_blank(monkeypatch, tmp_path):
    photo = tmp_path / "photo.jpg"
    Image.new("RGB", (32, 32), "white").save(photo)
    seen = {}
    def fake_complete(messages, **kwargs):
        seen.update(kwargs)
        return '{"summary":"One visible line","observations":["Front: line"],"scratch_count":1,"dent_count":"unknown","count_basis":"Front photo","next_view":"Left edge","uncertainty":"Dent not visible"}'
    monkeypatch.setattr(groq_provider, "complete", fake_complete)
    result = groq_provider.describe_selected_photos([(photo, "front")], "iPhone")
    assert seen["response_format"] == {"type": "json_object"}
    assert result["scratch_count"] == 1
    assert result["dent_count"] is None
