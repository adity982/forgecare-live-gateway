"""Create a plain-language report from the saved phone detector evaluation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def make_report(data: Path, run: Path) -> str:
    prepared = json.loads((data / "preparation.json").read_text(encoding="utf-8"))
    evaluated = json.loads((run / "evaluation.json").read_text(encoding="utf-8"))
    test = evaluated["test"]
    lines = ["# Phone exterior damage: what the held-out check found", "",
        "This is a local research model. A technician must check the phone and correct each proposed mark.", "",
        f"**Training:** {prepared['included']['train']} public phone photos; "
        f"**validation:** {prepared['included']['valid']}; **held-out check:** {test['photos']}.", "",
        "| Visible mark | Labeled regions in test photos | Found at the right location | Missed | Extra regions suggested |",
        "| --- | ---: | ---: | ---: | ---: |"]
    for name, counts in test["classes"].items():
        lines.append(f"| {name} | {counts['annotated_regions']} | {counts['localized_correctly']} | "
                     f"{counts['missed_regions']} | {counts['false_regions']} |")
    external_path = run / "external_screen_check.json"
    if external_path.exists():
        external = json.loads(external_path.read_text(encoding="utf-8"))["groups"]
        cracked = external["cracked screen"]
        intact = external["intact screen"]
        lines += ["", f"**Separate screen check:** the Broken alert appeared on {cracked['broken_alert_photos']} of {cracked['checked_photos']} cracked-screen photos and {intact['broken_alert_photos']} of {intact['checked_photos']} intact-screen photos from another public source. This tests screen-level alert behavior, not box location or dent and scratch counts."]
    lines += ["", "A proposed box is counted as found only when its type matches the annotation and it overlaps the labeled region by at least half. The confidence cutoff was chosen on validation photos before checking these test photos.", "",
        "**What a worker sees:** possible broken areas and scratches highlighted on each clear camera photo. Dent detection is disabled because it found none of the 14 labeled dent regions in the held-out check. The app drafts text that can be edited. The same mark may appear in several views, so photo suggestions are never summed into a physical-phone total. Only a technician enters confirmed counts.", "",
        "**What remains unproven:** the source test photos are almost all damaged, so this does not measure false alerts on clean phones. Dent examples are few; most test photos have one labeled mark. The source has no physical-phone IDs, and neither the iPhone 14 Plus nor realme narzo 60 has labeled ground truth in this study. The model cannot clear a phone or establish its repair need.", "",
        f"**Source:** {evaluated['source']} (CC BY 4.0).", "",
        f"**Prepared images:** `{data.resolve()}`", "",
        f"**Saved model, training record, and predictions:** `{run.resolve()}`", ""]
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    report = make_report(args.data, args.run)
    target = args.run / "BUSINESS_SUMMARY.md"
    target.write_text(report, encoding="utf-8")
    print(target)
