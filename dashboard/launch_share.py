"""Launch the Sorare dashboard with a temporary public Gradio share link."""

import os
import sys
import time
from pathlib import Path

import gradio as gr

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from dashboard.app import APP_CSS, build_dashboard


if __name__ == "__main__":
    port = int(os.getenv("SORARE_DASHBOARD_PORT", "7860"))
    app = build_dashboard()
    _, local_url, share_url = app.queue(default_concurrency_limit=2).launch(
        theme=gr.themes.Soft(),
        css=APP_CSS,
        server_name="127.0.0.1",
        server_port=port,
        share=True,
        prevent_thread_lock=True,
    )
    (ROOT_DIR / "dashboard" / "gradio_share_url.txt").write_text(
        f"local_url={local_url}\nshare_url={share_url}\n",
        encoding="utf-8",
    )
    while True:
        time.sleep(3600)
