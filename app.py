"""App shim: re-export the FastAPI application from the modular package.

This file is intentionally small so tools like gunicorn, Procfile and uvicorn
can continue to reference "app:app". The original, larger `app.py` has been
backed up to `app.py.bak`.
"""

import os

from moodsync.core import app  # re-export the FastAPI app created in moodsync.core
 
if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 5001))
    debug_mode = os.environ.get("FLASK_ENV", "production") == "development"
    log_level = "debug" if debug_mode else "info"

    uvicorn.run(app, host="0.0.0.0", port=port, reload=debug_mode, log_level=log_level)
