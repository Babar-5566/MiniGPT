# Troubleshooting

- **`python` or packages are missing:** activate `.venv` and run `python -m pip install -r requirements.txt`.
- **Checkpoint not found:** pass an existing `.pt` path; training checkpoints are under the experiment run directory.
- **Pytest cannot import `data`:** run from `C:\Tiny LLM`; `pytest.ini` adds the project root. Use the active environment's pytest.
- **Streamlit cannot import `inference`:** run `streamlit run app/streamlit_app.py` from the project root; the app bootstraps the root path.
- **CPU training is slow:** use `--config tiny` and a small `--steps` value for smoke checks.
- **Poor or repetitive generation:** this is expected for a small, short-trained, domain-specific model; try sampling controls or a better checkpoint.
- **Windows activation is blocked:** use `Set-ExecutionPolicy -Scope Process Bypass` in the current PowerShell session, or invoke `.venv\Scripts\python.exe` directly.
