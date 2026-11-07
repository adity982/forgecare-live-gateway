# Two-phone live inspection protocol

| Subject | Asset family / exact model | Official sources to index |
| --- | --- | --- |
| iPhone 14 Plus | `iphone-exterior` / `iPhone 14 Plus` | [iPhone 14 Plus repair manual](https://support.apple.com/en-us/100322), [battery procedure](https://support.apple.com/en-us/100346), [back-glass procedure](https://support.apple.com/en-us/100328), [broken glass safety](https://support.apple.com/en-us/100298) |
| realme narzo 60 5G | `realme-exterior` / `realme narzo 60 5G` | [model specifications](https://www.realme.com/in/realme-narzo-60/specs), [general support FAQ](https://www.realme.com/in/support/faq) |

The **Samsung Galaxy S24 FE is a camera**, not an inspection subject. The first two-camera mapping uses the Lenovo webcam (`0`) for the realme and the Samsung's live Wi-Fi MJPEG stream for the iPhone. Register the Samsung as `camera-samsung` after its actual stream URL is visible on the phone and reachable from the laptop. Use `iphone-physical-01` and `realme-physical-01` as distinct sample IDs. If both cameras inspect one physical phone from different angles, map both to that phone's asset and the same sample ID. Stop a worker before changing its asset, source, or sample ID. Do not use saved video as a live source.

Capture front, back and frame regions with a fixed background and adequate lighting. Avoid faces and unrelated workspace activity. Record only visible exterior labels: `none`, `minor`, `significant`, or `unclear`. `none` means no reviewed exterior finding in that region and frame; it does not mean the phone is safe or internally sound. Blurred or poorly lit frames require recapture or `unclear`.

Each model remains a separate training and retrieval scope. With one physical phone per model, grouped train/validation/test evaluation is impossible. The app can demonstrate live capture, quality routing, human labels, case prioritization, cited assistance, and fictional review exercises. The separate [VisA capsules research run](RESEARCH_AND_DATA.md) uses real manufacturing images but does not validate either phone. A deployable phone visual classifier needs additional distinct physical samples of each exact model and both label classes. The realme specifications page supports product-fact questions; it does not establish a repair procedure. Unknown repair questions should abstain.
