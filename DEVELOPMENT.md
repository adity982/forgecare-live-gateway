# Develop ForgeCare locally

This repository contains two parts. Render starts the Node gateway at the repository root. The Python inspection application served through that gateway lives in [`application/`](application/README.md) and runs on the connected laptop. Clone this repository to get the public source for both parts.

## Where to make changes

| Work | Files |
| --- | --- |
| Dashboard, forms, and workflow screens | [`application/forgecare/ui.py`](application/forgecare/ui.py), [`application/.streamlit/config.toml`](application/.streamlit/config.toml) |
| Users, assets, captures, reviews, cases, and API routes | [`application/forgecare/ops_api.py`](application/forgecare/ops_api.py), [`application/forgecare/db.py`](application/forgecare/db.py) |
| Camera capture, image quality, and visual features | [`application/forgecare/edge.py`](application/forgecare/edge.py), [`application/forgecare/browser_live_camera.py`](application/forgecare/browser_live_camera.py), [`application/forgecare/camera_setup.py`](application/forgecare/camera_setup.py), [`application/forgecare/vision.py`](application/forgecare/vision.py), [`application/forgecare/capture_summary.py`](application/forgecare/capture_summary.py) |
| Knowledge assistant and cited answers | [`application/forgecare/knowledge_api.py`](application/forgecare/knowledge_api.py), [`application/forgecare/groq_provider.py`](application/forgecare/groq_provider.py), [`application/forgecare/agent.py`](application/forgecare/agent.py) |
| Queue selection and priorities | [`application/forgecare/priority.py`](application/forgecare/priority.py), [`application/forgecare/sample_queue.py`](application/forgecare/sample_queue.py) |
| Federation and scheduled phone attempts | [`application/forgecare/federation.py`](application/forgecare/federation.py), [`application/forgecare/federated_client.py`](application/forgecare/federated_client.py), [`application/forgecare/phone_federation_schedule.py`](application/forgecare/phone_federation_schedule.py), [`application/run-phone-federation.ps1`](application/run-phone-federation.ps1) |
| Actual Flower automotive and casting research | [`application/scripts/train_autovi_federated.py`](application/scripts/train_autovi_federated.py), [`application/scripts/train_casting_research.py`](application/scripts/train_casting_research.py) |
| Fixed public URL and laptop bridge | [`server.js`](server.js), [`bridge_agent.py`](bridge_agent.py), [`ensure-bridge.ps1`](ensure-bridge.ps1), [`package.json`](package.json) |
| Automated checks and research methods | [`application/tests/`](application/tests/), [`application/scripts/`](application/scripts/), [`application/docs/`](application/docs/) |

The repository includes the remaining Python modules, launch scripts, pinned requirements, source documents, and research result summaries inside `application/`. Start with the linked file for the feature you want to change; use the neighboring tests to check the workflow.

## Run the Python application on Windows

Use 64-bit **Python 3.14.7**. The application launcher checks that exact version. In PyCharm, open this repository and set the project interpreter to `application\.venv\Scripts\python.exe` after creating the virtual environment below. Run the commands from a PowerShell terminal:

```powershell
git clone https://github.com/adity982/forgecare-live-gateway.git
cd forgecare-live-gateway\application
py -3.14 --version
py -3.14 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --retries 10 --timeout 120 --only-binary=:all: -r .\requirements.txt
powershell -ExecutionPolicy Bypass -File .\run.ps1
```

If PyCharm already created `application\.venv` with Python 3.14.7, skip the `venv` command. Confirm that the first command prints `Python 3.14.7`. The package installation includes a large PyTorch wheel; if a corporate network resets that download, retry with the shown timeout and retries or use an approved package mirror.

When the launcher reports its services ready, open <http://127.0.0.1:8501> on **this computer**. Create the first administrator and a site such as `demo-site`. The local database and captured photos are created in `%LOCALAPPDATA%\ForgeCareAI\data` by default; no account or capture history is included in the repository.

To index approved public guidance for the knowledge assistant after creating `demo-site`, run:

```powershell
.\.venv\Scripts\python.exe -m scripts.index_general_knowledge --site demo-site
```

Internet access to the approved source sites is needed for indexing. A Groq API key is optional for generated answers; keep it in your own environment, never in Git. The assistant can still show retrieved passages without it. To test and stop the application:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
powershell -ExecutionPolicy Bypass -File .\stop.ps1
```

The public checkout omits private captures, database records, downloaded training datasets, and two saved phone research weights. It can run the core UI and workflows with a fresh local database, but it cannot reproduce the saved weight-based phone checks without separately authorized artifacts. `requirements-vision.txt` is optional for that detector. `requirements-research.txt` documents an older separate research environment and should not be installed into the fixed Python 3.14.7 app environment.

## Run the gateway separately

For changes to the public gateway, install Node.js 20 or newer and run from the repository root:

```powershell
npm ci
npm start
```

The gateway listens on port 10000 by default. Its `/gateway-health` endpoint reports `laptopBridge: false` unless a separately configured, authenticated laptop bridge connects. Starting the gateway alone does not start the Python application. For ordinary feature work, use the Python app directly at `http://127.0.0.1:8501`.

## Contribute safely

Create a branch, commit your change, and open a pull request against `main`. Changes to gateway files such as `server.js` can redeploy Render after merging. Changes under `application/` update the public source snapshot; they do **not** change the Python process serving the live Render URL until the owner applies the same change on the host laptop and restarts its services. Keep credentials, real inspection images, downloaded datasets, and local databases out of commits.
