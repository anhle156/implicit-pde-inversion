"""Build the gallery animations from existing result CSVs (no training, no GPU).

    python make_gifs.py            # both
    python make_gifs.py scan       # river_scan.gif only
    python make_gifs.py ladder     # constraint_ladder.gif only

Axes are fixed across frames so only the data moves.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import LinearSegmentedColormap

SCEN = Path(__file__).resolve().parents[2]          # .../scenarios
OUT = SCEN / "gallary" / "animations"
SCAN_CSV = SCEN / "publishable_figure/Fig_JS1_JS2_PINN/results/contour/bed_inversion_26cs_long.csv"
LADDER_CSV = {
    "Parabolic channel": SCEN / "publishable_figure/_shared_inputs/parabolic_3scenarios.csv",
    "Bump (W-shape) channel": SCEN / "publishable_figure/_shared_inputs/bump_3scenarios.csv",
}

# ink / surface / series
INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb"
BLUE, ORANGE = "#2a78d6", "#eb6834"
WATER = "#cde2fb"
# one-hue sequential ramp: shallow = light, deep = dark
DEPTH_CMAP = LinearSegmentedColormap.from_list(
    "depth", ["#e6f0fc", "#9ec5f4", "#3987e5", "#184f95", "#0d366b"])

plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": "#c3c2b7", "axes.labelcolor": INK2,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.titlecolor": INK,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
})


def smoothstep(t):
    return t * t * (3 - 2 * t)


# --------------------------------------------------------------------------- #
# 1. River scan: sweep the 26 inverted cross sections from JS1 to JS2
# --------------------------------------------------------------------------- #
def river_scan(sub=3, fps=10, hold=15):
    d = pd.read_csv(SCAN_CSV)
    cs = sorted(d.cs.unique())
    grid = lambda col: np.stack([d.loc[d.cs == c, col].to_numpy() for c in cs])
    X, Y = grid("x_piv_m"), grid("y_target")            # plan-view coordinates
    S, H, U = grid("y_local_m"), grid("depth_pred"), grid("U_d_data")
    Z, Zs, WSE = grid("bed_elev_pred"), grid("bed_elev_survey"), grid("WSE")
    n = len(cs)

    fig = plt.figure(figsize=(10, 4.7), dpi=100)
    # column 2 is an empty spacer so the colorbar label clears the right-hand y labels
    gs = fig.add_gridspec(2, 4, width_ratios=[1.2, 0.04, 0.34, 1.15], height_ratios=[1, 1.35],
                          left=0.065, right=0.975, top=0.84, bottom=0.12, wspace=0.06, hspace=0.32)
    ax_map, cax = fig.add_subplot(gs[:, 0]), fig.add_subplot(gs[:, 1])
    ax_u, ax_z = fig.add_subplot(gs[0, 3]), fig.add_subplot(gs[1, 3])
    fig.suptitle("Riverbed recovered from surface velocity alone", x=0.065, ha="left",
                 fontsize=14, fontweight="bold", color=INK)
    fig.text(0.065, 0.895, "Physics-informed neural network (Shiono-Knight equation), 26 cross sections",
             color=INK2, fontsize=10)

    levels = np.linspace(0, np.ceil(H.max() * 10) / 10, 19)
    sm = plt.cm.ScalarMappable(cmap=DEPTH_CMAP, norm=plt.Normalize(levels[0], levels[-1]))
    cb = fig.colorbar(sm, cax=cax)
    cb.set_label("inferred depth (m)", color=INK2)
    cb.outline.set_visible(False)

    def row(A, k):                                       # fractional row k -> interpolated row
        i = min(int(np.floor(k)), n - 2)
        t = k - i
        return (1 - t) * A[i] + t * A[i + 1]

    ks = np.concatenate([np.linspace(0, n - 1, (n - 1) * sub + 1), np.full(hold, n - 1.0)])

    def draw(f):
        k = ks[f]
        for a in (ax_map, ax_u, ax_z):
            a.clear()
        # plan view, revealed up to the scan line
        i = int(np.floor(k))
        rows = lambda A: np.vstack([A[:i + 1], row(A, k)[None]]) if k > i else A[:i + 1]
        ax_map.plot(np.r_[X[:, 0], X[::-1, -1], X[0, 0]], np.r_[Y[:, 0], Y[::-1, -1], Y[0, 0]],
                    color="#c3c2b7", lw=1)
        if k > 0:
            ax_map.contourf(rows(X), rows(Y), rows(H), levels=levels, cmap=DEPTH_CMAP)
        xr, yr = row(X, k), row(Y, k)
        ax_map.plot(xr, yr, color=ORANGE, lw=2)
        ax_map.set(xlim=(X.min() - 0.3, X.max() + 0.3), ylim=(Y.min() - 0.3, Y.max() + 0.3),
                   xlabel="cross-stream  s (m)", ylabel="streamwise  Y (m)")
        ax_map.set_title("Plan view: bathymetry so far", loc="left", fontsize=10.5)
        ax_map.annotate("flow", xy=(X.max() + 0.05, 5.4), xytext=(X.max() + 0.05, 1.6),
                        ha="center", color=MUTED, fontsize=9,
                        arrowprops=dict(arrowstyle="->", color=MUTED, lw=1))
        # input: velocity across the current section
        s = row(S, k)
        ax_u.fill_between(s, 0, row(U, k), color=ORANGE, alpha=0.18, lw=0)
        ax_u.plot(s, row(U, k), color=ORANGE, lw=2)
        ax_u.set(xlim=(0, S.max()), ylim=(0, U.max() * 1.12), ylabel="U (m/s)")
        ax_u.set_title(f"Input: PIV velocity at Y = {yr[0]:.1f} m", loc="left", fontsize=10.5)
        ax_u.tick_params(labelbottom=False)
        ax_u.grid(axis="y", color=GRID, lw=0.8)
        # output: inverted bed under the water surface
        z, w = row(Z, k), row(WSE, k)
        ax_z.fill_between(s, z, w, color=WATER, lw=0)
        ax_z.plot(s, w, color=BLUE, lw=1, ls=":")
        ax_z.plot(s, z, color=BLUE, lw=2, label="PINN bed")
        j = int(round(k))
        if abs(k - j) < 1e-9 and np.isfinite(Zs[j]).all():
            ax_z.plot(s, Zs[j], color=INK, lw=2, label="surveyed bed")
        ax_z.set(xlim=(0, S.max()), ylim=(Z.min() - 0.08, WSE.max() + 0.1),
                 xlabel="distance across section (m)", ylabel="z (m)")
        ax_z.set_title("Output: inferred riverbed", loc="left", fontsize=10.5)
        ax_z.grid(axis="y", color=GRID, lw=0.8)
        ax_z.legend(loc="lower right", frameon=False, fontsize=9, labelcolor=INK2)

    anim = FuncAnimation(fig, draw, frames=len(ks), interval=1000 / fps)
    dst = OUT / "river_scan.gif"
    anim.save(dst, writer=PillowWriter(fps=fps))
    plt.close(fig)
    print(f"{dst.name}: {len(ks)} frames, {dst.stat().st_size / 1e6:.1f} MB")


# --------------------------------------------------------------------------- #
# 2. Constraint ladder: one prediction morphing as constraints are added
# --------------------------------------------------------------------------- #
STAGES = [
    ("depth_no_constraint", "1. Physics (PDE) only"),
    ("depth_discharge_only", "2. + known discharge Q"),
    ("depth_fixed_points", "3. + two depth soundings"),
]


def _fixed_point_idx():
    """Same indices as Fig_5_constraint_comparison/code/plot_fig5.py (= training scripts)."""
    import random
    random.seed(42)
    return {"Parabolic channel": [20, 80],
            "Bump (W-shape) channel": sorted(random.sample(range(20, 81), 2))}


def constraint_ladder(fps=12, morph=14, hold=16):
    fp_idx = _fixed_point_idx()
    data = {}
    for name, path in LADDER_CSV.items():
        d = pd.read_csv(path)
        d["x"] = d.distance - d.distance.min()
        data[name] = d

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6), dpi=100, sharey=True)
    fig.subplots_adjust(left=0.075, right=0.98, top=0.78, bottom=0.13, wspace=0.08)
    fig.suptitle("How much information does a PINN need to find the riverbed?", x=0.075,
                 ha="left", fontsize=14, fontweight="bold", color=INK)
    stage_txt = fig.text(0.075, 0.875, "", fontsize=12, color=BLUE, fontweight="bold")

    # timeline: hold stage 0, morph 0->1, hold 1, morph 1->2, long hold 2
    timeline = [(0, 0, 0.0)] * hold
    for a in range(len(STAGES) - 1):
        timeline += [(a, a + 1, smoothstep(t)) for t in np.linspace(0, 1, morph)]
        timeline += [(a + 1, a + 1, 0.0)] * (hold if a + 2 < len(STAGES) else hold * 2)

    def draw(f):
        a, b, t = timeline[f]
        shown = b if t > 0.5 else a
        stage_txt.set_text(STAGES[shown][1])
        for ax, (name, d) in zip(axes, data.items()):
            ax.clear()
            true = -d.true_depth
            pred = -((1 - t) * d[STAGES[a][0]] + t * d[STAGES[b][0]])
            for g in range(a + (1 if t > 0 else 0)):          # ghosts of finished stages
                ax.plot(d.x, -d[STAGES[g][0]], color="#9ec5f4", lw=1.5)
            ax.fill_between(d.x, true, 0, color=WATER, alpha=0.5, lw=0)
            ax.plot(d.x, true, color=INK, lw=2.5, label="true bed")
            ax.plot(d.x, pred, color=BLUE, lw=2.5, label="PINN prediction")
            if shown == 2:                                    # the two soundings given to the PINN
                i = fp_idx[name]
                ax.plot(d.x.iloc[i], true.iloc[i], "o", ms=9, mfc=SURFACE, mec=ORANGE, mew=2.5,
                        zorder=5, label="depth sounding")
            rmse = float(np.sqrt(np.mean((pred - true) ** 2)))
            ax.set_title(name, loc="left", fontsize=11)
            ax.text(0.98, 0.04, f"RMSE {rmse:.2f} m", transform=ax.transAxes, ha="right",
                    color=INK2, fontsize=11)
            ax.set(xlabel="distance across channel (m)", ylim=(-1.7, 0.08))
            ax.grid(axis="y", color=GRID, lw=0.8)
        axes[0].set_ylabel("depth below surface (m)")
        axes[1].legend(loc="center right", bbox_to_anchor=(0.985, 0.89),
                       bbox_transform=fig.transFigure, ncol=3, frameon=False,
                       fontsize=9.5, labelcolor=INK2, columnspacing=1.4, handlelength=1.6)

    anim = FuncAnimation(fig, draw, frames=len(timeline), interval=1000 / fps)
    dst = OUT / "constraint_ladder.gif"
    anim.save(dst, writer=PillowWriter(fps=fps))
    plt.close(fig)
    print(f"{dst.name}: {len(timeline)} frames, {dst.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("all", "scan"):
        river_scan()
    if which in ("all", "ladder"):
        constraint_ladder()
