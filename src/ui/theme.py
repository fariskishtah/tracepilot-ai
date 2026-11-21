CSS = """
:root{--bg:#0b1020;--panel:#111a33;--panel2:#0f1730;--line:#24305a;--blue:#3b82f6;--violet:#7c5cff;--text:#eaf0ff;--muted:#9fb0d0;--ok:#22c55e;--warn:#f59e0b;--bad:#ef4444}
.gradio-container{background:radial-gradient(1200px 500px at 10% -10%,#1a2350 0%,#0b1020 55%) !important;color:var(--text)!important;max-width:1280px!important}
.tp-hero{padding:22px 26px;border:1px solid var(--line);border-radius:16px;background:linear-gradient(120deg,#111a33,#18204a 60%,#271f5e);margin-bottom:12px}
.tp-hero h1{margin:0;font-size:30px;letter-spacing:.3px;color:#fff}.tp-hero p{margin:4px 0 0;color:var(--muted)}
.tp-hero .pill{display:inline-block;margin-top:10px;margin-right:6px;padding:3px 10px;border-radius:99px;background:#1b2652;border:1px solid var(--line);color:#c9d6ff;font-size:12px}
.timeline{position:relative;padding-left:6px}.tl{display:flex;gap:10px;margin:8px 0}
.tl .dot{flex:0 0 30px;height:30px;border-radius:50%;display:flex;align-items:center;justify-content:center;background:#1b2652;border:2px solid var(--blue);font-size:14px}
.tl.bad .dot{border-color:var(--bad)}.tl.warn .dot{border-color:var(--warn)}
.tl .card{flex:1;background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:8px 12px;min-width:0}
.tl .ttl{display:flex;justify-content:space-between;font-weight:600;color:#fff;font-size:14px}.tl .ttl small{color:var(--muted);font-weight:400}
.kv{display:flex;gap:8px;font-size:12px;margin-top:3px;align-items:flex-start}.kv span{flex:0 0 120px;color:var(--muted)}.kv code{background:#0b1230;color:#cfe0ff;padding:1px 6px;border-radius:6px;white-space:pre-wrap;word-break:break-word}
.steps{margin:4px 0 4px 18px;color:#dbe6ff;font-size:13px}.empty{padding:22px;color:var(--muted);border:1px dashed var(--line);border-radius:12px;text-align:center}
.wf{background:var(--panel2);border:1px solid var(--line);border-radius:12px;padding:8px}.legend{display:flex;gap:14px;font-size:12px;color:var(--muted);padding:4px 8px}.lg i{display:inline-block;width:10px;height:10px;border-radius:3px;margin-right:5px}
.answer-box{background:var(--panel);border:1px solid var(--line);border-radius:14px;padding:14px 16px}
button.primary,.primary{background:linear-gradient(90deg,var(--blue),var(--violet))!important;border:none!important;color:#fff!important}
@media (max-width:700px){.tp-hero h1{font-size:22px}.kv{flex-direction:column;gap:0}.kv span{flex:none}}
"""
