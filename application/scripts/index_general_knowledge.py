"""Index a reviewed, official source set for general ForgeCare questions.

Run from the project root with ``python -m scripts.index_general_knowledge``.
Re-running skips URLs that already have an active revision unless --refresh is used.
"""
from __future__ import annotations

import argparse

from forgecare.knowledge_api import DocIn, add_document, open_db


SOURCES = (
    ("iphone-general", "https://support.apple.com/en-us/101575"),
    ("android-general", "https://support.google.com/android/answer/7666942?hl=en"),
    ("android-general", "https://support.google.com/android/answer/7662405?hl=en"),
    ("manufacturing-general", "https://www.itl.nist.gov/div898/handbook/pmc/section3/pmc31.htm"),
    ("manufacturing-general", "https://www.itl.nist.gov/div898/handbook/pmc/section1/pmc16.htm"),
    ("manufacturing-general", "https://www.itl.nist.gov/div898/handbook/pmc/section2/pmc21.htm"),
    ("manufacturing-general", "https://www.nist.gov/metrology/metrological-traceability"),
    ("manufacturing-general", "https://www.nist.gov/digital-twins"),
    ("manufacturing-general", "https://www.nist.gov/mep/rise-artificial-intelligence-ai-us-manufacturing-text-only"),
    ("manufacturing-general", "https://www.nist.gov/news-events/news/2020/06/research-suggests-significant-benefits-investing-advanced-machinery"),
    ("manufacturing-general", "https://www.nist.gov/mep/successstories/2022/total-productive-maintenance-reduces-equipment-downtime-and-lost-capacity"),
    ("manufacturing-general", "https://www.lean.org/lexicon-terms/overall-equipment-effectiveness/"),
    ("automotive-general", "https://www.aiag.org/expertise-areas/quality/quality-core-tools"),
)

MODEL_SOURCES = (
    ("iphone-exterior", "iPhone 14 Plus", "https://support.apple.com/en-us/100322"),
    ("iphone-exterior", "iPhone 14 Plus", "https://support.apple.com/en-us/100346"),
    ("iphone-exterior", "iPhone 14 Plus", "https://support.apple.com/en-us/100328"),
    ("iphone-exterior", "iPhone 14 Plus", "https://support.apple.com/en-us/100329"),
    ("iphone-exterior", "iPhone 14 Plus", "https://support.apple.com/en-us/100299"),
    ("iphone-exterior", "iPhone 14 Plus", "https://support.apple.com/en-us/100298"),
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--site", default="demo-site")
    parser.add_argument("--refresh", action="store_true")
    options = parser.parse_args()
    with open_db() as db:
        active = {(row[0], row[1]) for row in db.execute(
            "SELECT family,url FROM documents WHERE site_id=? AND status='active'", (options.site,))}
    failures = []
    for family, model, url in [(family, "*", url) for family, url in SOURCES] + list(MODEL_SOURCES):
        if not options.refresh and (family, url) in active:
            print(f"SKIP {family}: {url}")
            continue
        try:
            result = add_document(DocIn(site_id=options.site, family=family, model=model,
                                        url=url, revision="official-web-2026-11"))
        except Exception as exc:
            failures.append((url, str(exc)))
            print(f"FAILED {family}: {url} ({type(exc).__name__})")
        else:
            print(f"INDEXED {family}: {result['passages']} passages · {result['title'][:65]}")
    if failures:
        raise SystemExit(f"{len(failures)} source(s) could not be indexed: " + ", ".join(url for url, _ in failures))


if __name__ == "__main__":
    main()
