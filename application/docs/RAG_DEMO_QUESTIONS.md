# Knowledge assistant demonstration

Choose **iPhone 14 Plus** under **Ask about asset**, leave **Question area** on **Automatic**, and ask these questions one at a time. Each returned answer is tied to an indexed approved page. The same local services power the fixed Render gateway.

| # | Question | Primary approved source |
|---|---|---|
| 1 | How do I check iPhone 14 Plus battery health? | [Apple Battery Health](https://support.apple.com/en-us/101575) |
| 2 | What should I do if the iPhone 14 Plus battery is visibly swollen? | [Apple Battery Safety](https://support.apple.com/en-us/100299) |
| 3 | How do I replace the iPhone 14 Plus front display? | [Apple iPhone 14 Plus Display](https://support.apple.com/en-us/100329) |
| 4 | How do I replace the iPhone 14 Plus back glass? | [Apple iPhone 14 Plus Back Glass](https://support.apple.com/en-us/100328) |
| 5 | How do I check an iPhone 14 Plus for cracked glass? | [Apple Broken Glass Safety](https://support.apple.com/en-us/100298) |
| 6 | What is a control chart in manufacturing? | [NIST Control Charts](https://www.itl.nist.gov/div898/handbook/pmc/section3/pmc31.htm) |
| 7 | What does FMEA mean in automotive quality? | [AIAG Quality Core Tools](https://www.aiag.org/expertise-areas/quality/quality-core-tools) |
| 8 | What is OEE in a factory? | [Lean Enterprise Institute OEE](https://www.lean.org/lexicon-terms/overall-equipment-effectiveness/) |
| 9 | What is metrological traceability? | [NIST Metrological Traceability](https://www.nist.gov/metrology/metrological-traceability) |
| 10 | What is a digital twin in manufacturing? | [NIST Digital Twins](https://www.nist.gov/digital-twins) |

These ten exact questions were checked against the laptop knowledge service on 7 November 2026. All ten returned source-linked answers. The first five use reviewed source summaries; the remaining five used Groq with per-claim citation validation. Free-provider availability can vary. If Groq cannot answer, the app tries a second Groq text model; if neither gives a verifiable answer, it shows related references without inventing one.

**Boundary check:** “What is the exact torque for a car engine bolt without a model or manual?” must not produce a specification. The assistant reports that an approved manual for that exact component is required and lists the indexed references it searched.

The Render address is a gateway to the same laptop dashboard and knowledge service. It does not run a second copy of the database or model. Keep the laptop and its bridge running for the Render link to work.
