"""Read-only, per-photo research suggestions from the locally trained detector."""
from __future__ import annotations

import cv2
import numpy as np
from time import perf_counter


CLASSES = ("broken area", "dent", "major scratch", "small scratch")
COLORS = ((49, 66, 228), (32, 138, 245), (38, 170, 87), (172, 105, 36))
DISPLAY_CLASSES = {0, 2, 3}  # Dent had zero correctly located held-out regions.


def load_model(weights_path: str):
    from ultralytics import YOLO
    return YOLO(weights_path)


def inspect(image_bytes: bytes, model, threshold: float) -> dict:
    image = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Cannot decode selected capture")
    started = perf_counter()
    result = model.predict(source=image, conf=max(.05, threshold), iou=.5, imgsz=416,
                           device="cpu", verbose=False)[0]
    runtime_ms = round((perf_counter() - started) * 1000, 1)
    overlay = image.copy()
    findings = []
    for box in result.boxes:
        klass = int(box.cls[0])
        if klass not in DISPLAY_CLASSES:
            continue
        confidence = float(box.conf[0])
        x1, y1, x2, y2 = (int(value) for value in box.xyxy[0].tolist())
        cv2.rectangle(overlay, (x1, y1), (x2, y2), COLORS[klass], 2)
        text = f"{CLASSES[klass]} {confidence:.2f}"
        cv2.putText(overlay, text, (x1, max(20, y1 - 7)), cv2.FONT_HERSHEY_SIMPLEX,
                    .55, COLORS[klass], 2, cv2.LINE_AA)
        findings.append({"class": CLASSES[klass], "confidence": confidence,
                         "box": [x1, y1, x2, y2]})
    counts = {name: sum(item["class"] == name for item in findings) for name in CLASSES}
    return {"overlay_rgb": cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB),
            "findings": findings, "counts": counts, "runtime_ms": runtime_ms}
