"""Shared helpers for the Chapter 4 figure scripts (TASK 96/97): style constants, the vintage-arm assertion,
and PDF+PNG output at 300 dpi into figures/. No data logic lives here."""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures"
INK, MUTED, GRID = "#1f2933", "#52606d", "#e4e7eb"
BLUE, ORANGE, GREEN, GREY, RED = "#2b6cb0", "#c05621", "#2f855a", "#9aa5b1", "#c53030"


def rel(p):
    return Path(p).resolve().relative_to(ROOT)


def require_vintage(arms_json):
    """Exit loudly unless the ARMS.json arm record carries gas_vintage_dispatch=True."""
    tag = json.loads(Path(arms_json).read_text())["arm_tag"]
    if "gas_vintage_dispatch=True" not in tag:
        sys.exit(f"FAIL: {arms_json} arm record does not carry gas_vintage_dispatch=True")


def require_vintage_tag(tag, where):
    if "gas_vintage_dispatch=True" not in tag:
        sys.exit(f"FAIL: {where} arm record does not carry gas_vintage_dispatch=True")


def style(ax, grid="y"):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8.5)
    if grid:
        ax.grid(True, axis=grid if grid in ("x", "y") else "both", color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)


def save(fig, stem, **kw):
    OUT.mkdir(exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{stem}.{ext}", dpi=300, **kw)
    print(f"wrote figures/{stem}.pdf and .png")
