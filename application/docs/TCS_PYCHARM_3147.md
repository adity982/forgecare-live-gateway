# Run ForgeCare in PyCharm on the TCS laptop

This portable copy targets **64-bit Python 3.14.7**. The operational application has three local processes: the Streamlit dashboard on port 8501, operations API on 8000, and knowledge API on 8001. The default database and captured photos are created under your own `%LOCALAPPDATA%\ForgeCareAI\data`, outside the project folder. They are not in the ZIP.

## First setup

1. Copy the ForgeCare ZIP to a folder you can write to on the TCS laptop. Extract it; do not run it from inside the ZIP.
2. In PowerShell, run `py -3.14 --version`. It must say `Python 3.14.7`. If not, install that exact 64-bit version using your organization's approved method, or point `FORGECARE_BASE_PYTHON` at a permitted `python.exe` for 3.14.7.
3. Open the extracted `ForgeCareAI-Release` folder as a project in PyCharm. In **Settings → Project → Python Interpreter**, add a new virtual environment at `<project>\.venv` with the 3.14.7 base interpreter. If PyCharm has already created it, the launcher will reuse it.
4. In PyCharm's PowerShell terminal, from the project root, run:

   ```powershell
   .\.venv\Scripts\python.exe --version
   .\.venv\Scripts\python.exe -m pip install --only-binary=:all: -r requirements.txt
   .\.venv\Scripts\python.exe -m pip install --no-deps -r requirements-vision.txt
   powershell -ExecutionPolicy Bypass -File .\run.ps1
   ```

   The second dependency command enables the locally saved phone-damage detector. It avoids installing a second OpenCV build over the tested headless build. If your network policy blocks PyPI, use your approved package mirror. `run.ps1` will stop with a version message if `.venv` is not 3.14.7.
5. Open `http://127.0.0.1:8501` in the TCS laptop browser. On a new machine, create the local administrator account on the first screen. The ZIP intentionally has no account, password, API key, camera history, or local operational database.
6. To enable the same cited iPhone 14 Plus battery and back-glass answers on this fresh laptop, run `.\.venv\Scripts\python.exe -m scripts.index_general_knowledge --site demo-site` after the services start. Use the Site ID you chose in step 5 if it differs from `demo-site`. This downloads approved public pages into the laptop's local knowledge database; internet access to those sites is required. Without indexing, the assistant cannot answer from those procedures.

## Inspect a phone

Under **Assets & setup**, register the phone model as an asset, then register each actual device under **Individual physical phones** with a stable, unique ID. Select the camera, stop its worker before changing its mapping, choose the model and the device ID, and enter the camera source. For an IP Webcam phone camera use its current `/video` URL while the phone and laptop are on the same permitted Wi-Fi network. Start the worker; **Live cameras** shows fresh frames only. In **Capture review**, save a clear technician-reviewed condition. The app opens an inspection work order for that photo. In **Inspection queue**, approve instructions, then finish the work order with a real finding. Physical inspection time is recorded only if the phone was actually handled.

The iPhone 14 Plus, realme narzo 60, and Samsung camera settings from the original laptop are **not copied**. The camera's earlier `192.168.*` address is a local-network address and must be replaced with the address shown by IP Webcam on the TCS network. An online host cannot access this local camera address.

## Local training and data

The ZIP includes source, saved model/research artifacts, reports, and tests. It does **not** contain private captures or the large downloaded public training datasets. The datasets used on the original laptop reside under `C:\Users\crazy\Documents\Codex\2026-11-03\ba\work\datasets`; see `docs\RESEARCH_AND_DATA.md` for source URLs and provenance. Several source datasets are noncommercial or have other redistribution terms, so fetch them separately if approved on the TCS laptop. The scheduled six-hour task is not installed by opening the ZIP; run `install-phone-federation-task.ps1` from PowerShell when you want it. Until it sees enough *distinct* physically reviewed phones, it records a skipped attempt instead of a phone model.

For optional Groq knowledge answers and image drafts, set `GROQ_API_KEY` as a Windows user environment variable using your approved secret-management method before launching the app. The local detector and ordinary case workflow work without it. Do not put credentials in a project file or ZIP.

## Stop and test

Run `.\stop.ps1` from the project folder to stop the three services. Run `.\.venv\Scripts\python.exe -m pytest -q` for the automated checks. If the dashboard does not start, look in `%LOCALAPPDATA%\ForgeCareAI\data\logs` for the corresponding service log.
