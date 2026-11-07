# Private local bundle

`ForgeCareAI-Private-Local-Complete-Python-3.14.7.zip` is the self-contained **project and data** transfer. It includes this application's source, saved research models and evaluation artifacts, downloaded training datasets, cached local language and embedding models, Windows dependency wheels, the official Python 3.14.7 Windows installer, a SQLite snapshot of the current cases and audit history, captured images, the internal service key, and the Groq and Roboflow API keys. The ZIP also contains `PRIVATE_LOGIN.txt` with a newly generated password for the copied admin account. The current laptop's live database and password are not changed by packaging.

This archive contains usable credentials and inspection images. Keep it in a private location and do not upload or share it as a public repository or demo download. Several bundled public datasets have noncommercial or other redistribution terms; `docs/RESEARCH_AND_DATA.md` lists their sources and licenses. The Hugging Face deployment token is not required for local execution and is not included.

## Run after extraction

1. Extract the ZIP to a writable folder. Its top-level folder is `ForgeCareAI-Release`. Open that folder in PyCharm.
2. Install **64-bit Python 3.14.7** if needed. The official installer is bundled at `setup\python-3.14.7-amd64.exe`; use your organization's approved installation method. In PowerShell, check `py -3.14 --version`. It must say `Python 3.14.7`.
3. From the extracted project root, run `powershell -ExecutionPolicy Bypass -File .\run-private.ps1`. On first use it creates `.venv` and installs the pinned main and phone-detector packages from `setup\wheelhouse` without contacting PyPI. Later runs reuse the environment.
4. Open `http://127.0.0.1:8501`. Use the username and newly generated password in `PRIVATE_LOGIN.txt`. You can change that password in **Administration** after signing in.
5. Use `powershell -ExecutionPolicy Bypass -File .\stop-private.ps1` to stop the three local services.

`run-private.ps1` keeps the copied database and images in `private_data` inside the extracted folder. It reads `private_config.json` for the Groq and Roboflow keys and points model loading at the included `model_cache`; no Windows user environment setup is required. The local detector, queue, captured reviews, saved research artifacts, and cached text and embedding models run on the laptop. Groq answers and visual drafts call Groq over the internet when used, and are subject to that account's access and quota. The included wheelhouse removes the need for PyPI during the initial setup on 64-bit Windows.

The downloaded research images are in `datasets`. Research manifests and copied capture records have portable paths relative to the extracted project, so they remain usable if the folder moves. If you move the project while ForgeCare is running, stop it first, move the whole folder, and restart it from the new path.

The saved Samsung IP Webcam address belongs to the original Wi-Fi network. On another network, open **Assets & setup**, stop the camera worker, enter the camera's current `/video` address, save the mapping, and restart the worker. Previous captures remain available even while the camera is offline.

To enable the six-hour local phone federation attempt, run `powershell -ExecutionPolicy Bypass -File .\install-phone-federation-task.ps1` after first setup. The scheduled task is an operating-system setting and cannot be installed simply by extracting the ZIP. It uses the copied `private_data` and still skips training until enough distinct physically reviewed phones exist.
