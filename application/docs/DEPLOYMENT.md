# Deployment notes

The local demonstration binds all services to `127.0.0.1`. On separate hosts, put the operations and knowledge APIs behind protected transport and client authentication, keep the service key in a managed secret store, and restrict filesystem permissions for `data/`. Do not expose the local ports directly to a network.

The edge device only needs a camera adapter and a validated small visual model. Raspberry Pi integration has not been tested. Keep LLM inference on Server 2. Protect raw images and retention according to site policy. Model parameter sharing does not guarantee privacy; secure aggregation and differential privacy require a separate design and evaluation.

An edge-only host can install `requirements-edge.txt`, set `FORGECARE_OPS_URL` to its protected operations endpoint, set `FORGECARE_SERVICE_KEY` to the shared service secret, and run `python -m forgecare.edge --camera CAMERA_ID`. This adapter has not been run on Raspberry Pi hardware; verify camera drivers, wheel availability, network security, and inference latency before calling it supported.

Scale experiments should compare latency, memory and throughput before cloud or Pi deployment. Azure and quantization are optional architecture exercises, not prerequisites for the local demo.
