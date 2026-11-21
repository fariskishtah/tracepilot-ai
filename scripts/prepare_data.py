"""Prepare the TracePilot AI knowledge base.

1. The 31 base documents (4 courses) are original CC BY 4.0 material written by the
   project owner for CourseAtlas AI / ReGround AI. They were copied (read-only) from the
   ReGround AI corpus into data/documents/.
2. This script adds NEW TracePilot-specific documents (original, CC BY 4.0) that
   create genuine multi-step needs: an overfitting-mitigation guide, a regularization
   experiment CSV, a product price CSV (skewed, for mean-vs-median questions) and a
   robust-statistics note.
3. It (re)writes data/manifests/sources.json with SHA-256 hashes and provenance.

Run:  python scripts/prepare_data.py
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "data" / "documents"
ATTR = "Faris Mohamed Kishtah"

REG_CSV = """run_id,model,dataset,technique,train_accuracy,val_accuracy,epochs,notes
R01,MLP (2 hidden layers),image-tiny,none (baseline),0.998,0.812,60,Memorizes training set; large train-val gap
R02,MLP (2 hidden layers),image-tiny,L2 weight decay 1e-3,0.951,0.874,60,Gap shrinks; weights shrunk
R03,MLP (2 hidden layers),image-tiny,dropout p=0.3,0.943,0.881,60,Stochastic unit removal at training time
R04,MLP (2 hidden layers),image-tiny,early stopping patience=5,0.927,0.866,23,Stopped at epoch 23 of 60 saving compute
R05,MLP (2 hidden layers),image-tiny,L2 + early stopping,0.935,0.889,27,Best validation accuracy of the image-tiny runs
R06,MLP (2 hidden layers),image-tiny,data augmentation,0.962,0.893,60,Highest validation accuracy but needs extra preprocessing
R07,Gradient Boosting,tabular-churn,none (baseline),0.991,0.842,300,Deep trees overfit
R08,Gradient Boosting,tabular-churn,shrinkage 0.05 + max_depth 3,0.902,0.861,300,Shallower trees reduce variance
R09,Gradient Boosting,tabular-churn,early stopping rounds=20,0.918,0.867,118,Stopped at 118 of 300 rounds
"""

PRICE_HEADER = "product_id,category,product_name,price_usd,units_sold\n"

MITIGATION_TXT = """OVERFITTING MITIGATION GUIDE - TracePilot demo course pack (Machine Learning Fundamentals)
Original educational material, CC BY 4.0, (c) Faris Mohamed Kishtah.

1. What overfitting means in practice
A model overfits when its training accuracy is far higher than its validation accuracy. The gap
between the two (train accuracy minus validation accuracy) is the simplest warning sign.

2. L2 regularization (weight decay)
L2 regularization adds a penalty proportional to the sum of squared weights to the loss, which
shrinks weights toward zero without usually making them exactly zero. Advantages: it is smooth,
works with gradient-based optimizers, and keeps all features. Disadvantages: it introduces a
strength hyperparameter (lambda or weight decay) that must be tuned with cross-validation, and it
does not remove irrelevant features. It increases bias slightly in exchange for lower variance.

3. Early stopping
Early stopping monitors validation loss during training and stops when it has not improved for a
fixed number of epochs (the patience); the best checkpoint is restored. Advantages: it needs no
change to the model, saves training compute, and the stopping epoch acts as an implicit
regularizer. Disadvantages: it requires a held-out validation set that is then not used for
training, results are sensitive to the patience value, and noisy validation curves can trigger a
premature stop.

4. Dropout
Dropout randomly zeroes a fraction p of hidden units at each training step so units cannot
co-adapt. It is applied only during training. It is mainly used for neural networks.

5. Choosing between methods
L2 and early stopping are complementary and are often combined. Early stopping is cheapest;
L2 gives a smoother, more reproducible solution. Compare methods by validation accuracy, never by
training accuracy.
"""

ROBUST_TXT = """ROBUST SUMMARIES: WHY THE MEAN AND THE MEDIAN DIFFER
Original educational material, CC BY 4.0, (c) Faris Mohamed Kishtah. TracePilot demo course pack.

The mean is the sum of the values divided by their count, so every value, including extreme ones,
pulls it. The median is the middle value of the sorted data and ignores how far the extremes lie.
When a distribution is right-skewed (a long tail of a few very large values, as with prices or
incomes) a few large values drag the mean upward while the median stays near the typical value, so
the mean exceeds the median. When the data are symmetric the two are close. A large gap between the
mean and the median is therefore a quick diagnostic for skew or outliers. For skewed data, report
the median (or both) because it better describes a typical observation.
"""


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def make_prices() -> str:
    rng = np.random.default_rng(42)
    cats = ["Accessories", "Audio", "Computing", "Home"]
    base = {"Accessories": 18, "Audio": 60, "Computing": 240, "Home": 45}
    rows = []
    for i in range(1, 41):
        c = cats[(i - 1) % 4]
        price = round(float(base[c] * rng.lognormal(0, 0.55)), 2)
        rows.append((f"P{i:03d}", c, f"{c} item {i}", price, int(rng.integers(5, 120))))
    return PRICE_HEADER + "\n".join(f"{a},{b},{c},{d},{e}" for a, b, c, d, e in rows) + "\n"


def make_pdf(path: Path, title: str, paragraphs: list[tuple[str, str]]) -> None:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer

    st = getSampleStyleSheet()

    def footer(c, d):
        c.setFont("Helvetica", 8)
        c.drawString(50, 800, f"TracePilot Demo Course Pack | {title}")
        c.drawString(50, 788, f"Page {d.page}")
        c.drawString(50, 776, "Original educational material, CC BY 4.0")

    story = []
    for i, (h, body) in enumerate(paragraphs):
        if i:
            story.append(PageBreak())
        story += [Paragraph(h, st["Heading2"]), Spacer(1, 8), Paragraph(body, st["BodyText"])]
    SimpleDocTemplate(str(path), pagesize=A4, topMargin=90, title=title, author=ATTR).build(
        story, onFirstPage=footer, onLaterPages=footer
    )


def main() -> None:
    (DOCS / "ml" / "ml_regularization_experiments.csv").write_text(REG_CSV)
    (DOCS / "data" / "product_prices.csv").write_text(make_prices())
    (DOCS / "statistics" / "statistics_robust_summaries.txt").write_text(ROBUST_TXT)
    (DOCS / "ml" / "ml_overfitting_mitigation_guide.txt").write_text(MITIGATION_TXT)
    # A PDF version carries real page numbers for page-level citations.
    secs = [s.strip() for s in MITIGATION_TXT.split("\n\n") if s.strip()]
    heads = [("Overfitting mitigation guide", " ".join(secs[:3])),
             ("L2 regularization and early stopping", " ".join(secs[3:5])),
             ("Dropout and choosing between methods", " ".join(secs[5:]))]
    make_pdf(DOCS / "ml" / "ml_overfitting_mitigation_guide.pdf", "Overfitting Mitigation", heads)
    (DOCS / "ml" / "ml_overfitting_mitigation_guide.txt").unlink()

    old = {(m["course_id"], m["filename"]): m for m in json.loads((ROOT / "data/manifests/sources.json").read_text())} \
        if (ROOT / "data/manifests/sources.json").exists() else {}
    new_files = {
        "ml_regularization_experiments.csv", "product_prices.csv",
        "statistics_robust_summaries.txt", "ml_overfitting_mitigation_guide.pdf",
    }
    out = []
    for p in sorted(DOCS.glob("*/*")):
        course = p.parent.name
        m = old.get((course, p.name), {})
        is_new = p.name in new_files
        out.append({
            "course_id": course, "filename": p.name, "sha256": sha256(p), "bytes": p.stat().st_size,
            "origin": "tracepilot-ai" if is_new else m.get("origin", "reground-ai"),
            "license": "CC BY 4.0",
            "attribution": f"{ATTR}, " + ("TracePilot AI" if is_new else "CourseAtlas AI / ReGround AI"),
            "challenge": "TracePilot multi-step/tool material" if is_new else m.get("challenge", "course material"),
        })
    (ROOT / "data" / "manifests").mkdir(exist_ok=True)
    (ROOT / "data" / "manifests" / "sources.json").write_text(json.dumps(out, indent=2))
    print(f"{len(out)} documents in manifest")


if __name__ == "__main__":
    main()
