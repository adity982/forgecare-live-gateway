"""Clearly fictional review exercises. Never insert these rows into operations data."""

SAMPLE_REVIEWS = [
    {"id": "PRACTICE-01", "subject": "Phone exterior", "views": "Front and right edge",
     "image_quality": "Two clear views", "draft": "Two photos passed focus and brightness checks. No approved phone defect detector is available.",
     "technician_review": "A shallow mark is visible along the right edge. No dent can be confirmed from these views.",
     "grade": "minor", "scratch_count": 1, "dent_count": None,
     "next_step": "Ask for a close-up before confirming whether the mark is a scratch."},
    {"id": "PRACTICE-02", "subject": "Phone exterior", "views": "Front only",
     "image_quality": "Image is dark", "draft": "The image does not pass the brightness check.",
     "technician_review": "Condition cannot be judged from this image.",
     "grade": "unclear", "scratch_count": None, "dent_count": None,
     "next_step": "Improve lighting and recapture. Do not open a condition case from this photo."},
    {"id": "PRACTICE-03", "subject": "Metal part", "views": "Top and detail",
     "image_quality": "Two clear views", "draft": "The research model would require validation for this exact part before use.",
     "technician_review": "The top view shows a visible surface discontinuity; the detail view needs human confirmation.",
     "grade": "significant", "scratch_count": None, "dent_count": None,
     "next_step": "Send the marked part for supervisor review and follow the approved procedure."},
]
