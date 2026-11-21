"""Drive the REAL Gradio app in a REAL browser (Playwright/Chromium), capture screenshots, console errors and a JSON report.
Usage:  python scripts/browser_verify.py [--port 7871] [--shots artifacts/screenshots]"""
import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MULTI = ("Find the definition of overfitting, retrieve two mitigation techniques, compare them, and calculate the difference between the validation "
         "accuracies of run R02 and run R04 in the regularization experiments dataset.")


def wait_http(url, t=120):
    t0 = time.time()
    while time.time() - t0 < t:
        try:
            urllib.request.urlopen(url, timeout=2)
            return True
        except Exception:
            time.sleep(1)
    return False


def run(port=7871, shots: Path | None = None, headless=True, start_server=True) -> dict:
    from playwright.sync_api import sync_playwright
    shots = Path(shots or ROOT / "artifacts" / "screenshots")
    shots.mkdir(parents=True, exist_ok=True)
    rep = {"steps": [], "failures": [], "console_errors": [], "failed_requests": [], "screenshots": []}
    proc = None
    log = Path(tempfile.mkdtemp()) / "server.log"
    if start_server:
        env = dict(os.environ, TRACEPILOT_PORT=str(port))
        proc = subprocess.Popen([sys.executable, "-W", "ignore", str(ROOT / "app.py")], cwd=ROOT, env=env, stdout=open(log, "w"), stderr=subprocess.STDOUT)
    url = f"http://127.0.0.1:{port}"
    try:
        if not wait_http(url):
            rep["failures"].append("server did not start")
            return rep

        def step(name, ok, detail=""):
            rep["steps"].append({"step": name, "ok": bool(ok), "detail": str(detail)[:300]})
            if not ok:
                rep["failures"].append(f"{name}: {detail}")

        def shot(page, name):
            p = shots / f"{name}.png"
            page.screenshot(path=str(p), full_page=False)
            rep["screenshots"].append(p.name)

        with sync_playwright() as pw:
            b = pw.chromium.launch(headless=headless)
            ctx = b.new_context(viewport={"width": 1440, "height": 1000})
            page = ctx.new_page()
            page.on("console", lambda m: rep["console_errors"].append(m.text[:200]) if m.type == "error" else None)
            page.on("requestfailed", lambda r: rep["failed_requests"].append(r.url[:120]))
            page.goto(url, wait_until="domcontentloaded", timeout=60000)
            page.get_by_text("Agentic RAG & Intelligent Knowledge Orchestration").first.wait_for(timeout=60000)
            time.sleep(1.5)
            step("page loads with title", "TracePilot" in page.title() or page.get_by_text("TracePilot AI").first.is_visible(), page.title())
            shot(page, "01_main_agentic_qa")

            # ---- mobile
            m = b.new_context(viewport={"width": 390, "height": 844}, device_scale_factor=2, is_mobile=True)
            mp = m.new_page()
            mp.goto(url, wait_until="domcontentloaded", timeout=60000)
            mp.get_by_text("Agentic RAG & Intelligent Knowledge Orchestration").first.wait_for(timeout=60000)
            time.sleep(1.5)
            sw = mp.evaluate("document.documentElement.scrollWidth")
            step("mobile layout has no horizontal overflow", sw <= 392, f"scrollWidth={sw}")
            mp.screenshot(path=str(shots / "12_mobile_layout.png"))
            rep["screenshots"].append("12_mobile_layout.png")

            # ---- multi-step run with live trace
            page.get_by_label("Your question").fill(MULTI)
            page.get_by_role("button", name="Run agent").click()
            try:
                page.get_by_text("Done in").first.wait_for(timeout=240000)
                step("multi-step agent run completes", True)
            except Exception as e:
                step("multi-step agent run completes", False, e)
            txt = page.locator("body").inner_text()
            step("trace shows planned tools", "tool selected" in txt and "search_knowledge_base" in txt)
            step("answer has citations", "[S" in txt or "[D" in txt)
            step("tool sequence reported", "Tool sequence" in txt)
            shot(page, "02_multistep_answer_and_live_trace")
            page.get_by_text("Live execution trace").first.scroll_into_view_if_needed()
            shot(page, "03_live_execution_trace")

            def tab(name):
                page.get_by_role("tab", name=name).click()
                time.sleep(0.8)

            tab("Execution Trace")
            page.get_by_role("button", name="Show last request").click()
            time.sleep(1.5)
            step("execution trace tab renders plan", "plan" in page.locator("body").inner_text().lower())
            shot(page, "04_execution_trace_tab")

            tab("Tool Explorer")
            page.get_by_label("Tool", exact=True).first.click()
            page.get_by_role("option", name="calculate_expression").click()
            time.sleep(0.5)
            editor = page.locator(".cm-content").first
            editor.click()
            page.keyboard.press("Meta+A")
            page.keyboard.type('{"expression": "percent_of(15, 240) + 2**3"}')
            page.get_by_role("button", name="Execute safely").click()
            time.sleep(2.5)
            step("tool explorer runs calculator", "44" in page.locator("body").inner_text())
            shot(page, "05_tool_explorer_calculator")

            tab("Knowledge Library")
            body = page.locator("body").inner_text()
            step("knowledge library lists indexed docs", "ml_lecture_notes" in body and "product_prices" in body)
            up = Path(tempfile.mkdtemp()) / "browser_upload.txt"
            up.write_text("Gradient clipping limits the norm of gradients to stabilise recurrent network training. " * 4)
            with page.expect_file_chooser() as fc:
                page.get_by_text("Click to Upload").first.click()
            fc.value.set_files(str(up))
            time.sleep(1.5)
            page.get_by_role("button", name="Upload & index").click()
            time.sleep(6)
            body = page.locator("body").inner_text()
            step("upload indexes a new document", "browser_upload" in body and "chunks)" in body, "")
            shot(page, "06_knowledge_library_upload")

            tab("Evidence Inspector")
            page.get_by_role("button", name="Load evidence of last request").click()
            time.sleep(1.5)
            step("evidence inspector shows original passage with provenance", "Original document passage" in page.locator("body").inner_text() or "Deterministic" in page.locator("body").inner_text())
            shot(page, "07_evidence_inspector")

            tab("Agent Workflow")
            page.get_by_role("button", name="Render last execution").click()
            time.sleep(1.5)
            step("workflow svg rendered from real run", page.locator("svg").count() > 0 and "Synthesis" in page.locator("body").inner_text())
            shot(page, "08_agent_workflow")

            tab("Standard vs Agentic RAG")
            page.get_by_label("Question (same question, same model, same retriever)").fill("Compare supervised and unsupervised learning.")
            page.get_by_role("button", name="Run both").click()
            try:
                page.get_by_text("citation support rate").first.wait_for(timeout=240000)
                step("standard vs agentic comparison runs", True)
            except Exception as e:
                step("standard vs agentic comparison runs", False, e)
            shot(page, "09_standard_vs_agentic")

            tab("Analytics")
            page.get_by_role("button", name="Refresh analytics").click()
            time.sleep(2)
            step("analytics computed from recorded traces", "Requests:" in page.locator("body").inner_text())
            shot(page, "10_analytics_dashboard")

            tab("Failure Analysis")
            time.sleep(0.5)
            shot(page, "11_failure_analysis")

            b.close()
        # a real error is an error in the app's own JS; ignore benign third-party/network noise but report everything
        rep["server_log_tail"] = log.read_text()[-1200:] if log.exists() else ""
        step("server log has no tracebacks", "Traceback" not in rep["server_log_tail"], rep["server_log_tail"][-200:])
    finally:
        if proc:
            proc.terminate()
            try:
                proc.wait(10)
            except Exception:
                proc.kill()
    return rep


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=7871)
    ap.add_argument("--shots")
    a = ap.parse_args()
    r = run(a.port, a.shots)
    out = ROOT / "artifacts" / "metrics" / "browser_verification.json"
    out.write_text(json.dumps(r, indent=2))
    print(json.dumps({k: r[k] for k in ("failures", "console_errors", "failed_requests", "screenshots")}, indent=1))
    print(f"{sum(s['ok'] for s in r['steps'])}/{len(r['steps'])} steps passed")
