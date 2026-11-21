"""TracePilot AI - launch the Gradio Studio:  python app.py"""
import os

from src.config import cfg
from src.service import Service
from src.ui.app_ui import build_app

if __name__ == "__main__":
    s = cfg("app")["server"]
    build_app(Service()).queue(default_concurrency_limit=1).launch(server_name=s["host"], server_port=int(os.environ.get("TRACEPILOT_PORT", s["port"])), show_api=False)
