
# Developer & Deployment Guide (Consolidated)

This single file replaces the previous separate development, deployment and contributing documents. It contains the essential steps to run, develop, and deploy the MoodSync AI Service locally and in production.

## Quick start (recommended)

 - Use the Python runtime specified in `runtime.txt` (python-3.10.14).

 - Create a Python 3.10 virtual environment named `venv310` and install requirements:

 ```cmd
 C:\> C:\Path\To\python3.10.exe -m venv "%CD%\\venv310"
 C:\> .\venv310\Scripts\activate
 (venv310) C:\> python -m pip install --upgrade pip setuptools wheel
 (venv310) C:\> python -m pip install -r requirements.txt --no-cache-dir
 ```

 Quick import check:

 ```cmd
 (venv310) C:\> python -c "import importlib; importlib.import_module('app'); print('APP IMPORT OK')"
 ```

 Run locally (dev):

 ```cmd
 (venv310) C:\> uvicorn app:app --reload --port 5001
 ```

## Files of interest

 - `app.py` — shim that re-exports the FastAPI app from `moodsync.core` (keeps `app:app` import path stable).

 - `moodsync/` — application package; main code lives here.

 - `requirements.txt`, `runtime.txt` — dependency and runtime pinning.

 - `render.yaml`, `Procfile` — deployment configuration for Render (if used).

## Deployment notes (Render / Docker)

 - Preferred approach: force Python 3.10 via `runtime.txt` or deploy with Docker that uses Python 3.10.

 - If the target platform forces Python 3.13, consider one of:

   - Use a requirements file targeted at Python 3.13 (if available)

   - Deploy via Docker with our `Dockerfile` which pins Python 3.10

   - Use an alternative host that supports explicit Python version selection

### Render-specific tips

 - Render may ignore `runtime.txt` in some cases. Prefer Docker or confirm Render's build command sets the right interpreter.

 - Ensure `GEMINI_API_KEY` and other env vars are set in the platform environment.

## Troubleshooting

 - If pip tries to compile pydantic-core (Rust) or httptools (C) and fails on Windows, use Python 3.10 and the pinned requirements to get prebuilt wheels.

 - For authentication or rate-limit issues, check the `GEMINI_API_KEY` and monitor logs for `rate_limit` errors.

## Contributing (short)

 - Fork, branch, make changes, ensure imports pass and add tests.

 - Keep changes small and focused. Run `python -m pytest` if tests are present.

## Emergency fallback

 - If deployments fail due to platform Python mismatch, use the provided `Dockerfile` or create a minimal `requirements.txt` with older package versions as a temporary fallback.

---

For full examples and legacy details (previous separate docs were merged here), use this as the single source of truth. If you need parts of the older documentation restored as separate files, I can extract them on request.
