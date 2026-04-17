"""
utils/plot.py — NeuroSnake Live Training Curve
===============================================
Provides a non-blocking Matplotlib figure that updates in real-time
during DQN training, plus a save() method to export the final curve.

Usage (from training/train.py):
    from utils.plot import plot
    plot(scores, mean_scores)                         # live update
    plot(scores, mean_scores, filename="curve.png", save_only=True)  # export
"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np


# ─── Colour palette (mirrors the game's deep-navy aesthetic) ─────────────────
_BG_OUTER   = "#0d0d1a"   # figure background
_BG_AXES    = "#12122a"   # axes background
_CLR_SCORE  = "#00dc96"   # teal-green — score line
_CLR_MEAN   = "#ffbe00"   # gold       — rolling-mean line
_CLR_GRID   = "#1e1e40"   # subtle grid lines
_CLR_TEXT   = "#c8c8ff"   # lavender text / labels
_CLR_SPINE  = "#2a2a55"   # axis spines


class LivePlot:
    """
    Non-blocking Matplotlib learning curve that updates during training.

    Attributes
    ----------
    fig : matplotlib.figure.Figure
    ax  : matplotlib.axes.Axes
    """

    def __init__(self):
        # Switch to an interactive backend if possible; fall back gracefully.
        try:
            matplotlib.use("TkAgg")
        except Exception:
            pass  # headless / already set — not critical

        plt.ion()   # enable non-blocking mode

        self.fig, self.ax = plt.subplots(figsize=(10, 5))
        self.fig.patch.set_facecolor(_BG_OUTER)
        self.fig.canvas.manager.set_window_title("NeuroSnake — Training Progress")

        self._apply_style()

        # Placeholder line objects (updated each call instead of re-drawing)
        (self._line_score,) = self.ax.plot([], [], color=_CLR_SCORE,
                                           linewidth=1.5, alpha=0.75,
                                           label="Score / episode")
        (self._line_mean,)  = self.ax.plot([], [], color=_CLR_MEAN,
                                           linewidth=2.5, label="Mean score (last 100)")

        self.ax.legend(facecolor=_BG_AXES, edgecolor=_CLR_SPINE,
                       labelcolor=_CLR_TEXT, fontsize=10)

        plt.tight_layout(pad=2.0)
        plt.show(block=False)

    # ──────────────────────────────────────────────────────────────────────────
    def _apply_style(self):
        """Apply the dark-navy style to the axes."""
        ax = self.ax
        ax.set_facecolor(_BG_AXES)

        for spine in ax.spines.values():
            spine.set_edgecolor(_CLR_SPINE)

        ax.tick_params(colors=_CLR_TEXT, labelsize=9)
        ax.xaxis.label.set_color(_CLR_TEXT)
        ax.yaxis.label.set_color(_CLR_TEXT)
        ax.title.set_color(_CLR_TEXT)

        ax.set_xlabel("Episode", fontsize=11)
        ax.set_ylabel("Score",   fontsize=11)
        ax.set_title("NeuroSnake — DQN Training Progress", fontsize=13, fontweight="bold")
        ax.grid(True, color=_CLR_GRID, linestyle="--", linewidth=0.6, alpha=0.8)

    # ──────────────────────────────────────────────────────────────────────────
    def update(self, scores: list, mean_scores: list):
        """
        Refresh the plot with the latest data.

        Parameters
        ----------
        scores      : list of per-episode scores
        mean_scores : list of rolling-mean scores (same length as scores)
        """
        if not scores:
            return

        episodes = list(range(1, len(scores) + 1))

        self._line_score.set_data(episodes, scores)
        self._line_mean.set_data(episodes, mean_scores)

        # Auto-rescale axes
        self.ax.relim()
        self.ax.autoscale_view()

        # Annotate the latest values on the right edge
        self._annotate_latest(episodes, scores, mean_scores)

        self.fig.canvas.draw()
        plt.pause(0.001)   # yield control to the GUI event loop briefly

    # ──────────────────────────────────────────────────────────────────────────
    def _annotate_latest(self, episodes, scores, mean_scores):
        """Show the current score and rolling mean as text annotations."""
        # Remove previous annotations to avoid stacking
        for txt in getattr(self, "_annotations", []):
            try:
                txt.remove()
            except Exception:
                pass

        self._annotations = []

        last_ep    = episodes[-1]
        last_score = scores[-1]
        last_mean  = mean_scores[-1]

        ann1 = self.ax.annotate(
            f"{last_score}",
            xy=(last_ep, last_score),
            xytext=(6, 4), textcoords="offset points",
            color=_CLR_SCORE, fontsize=8, fontweight="bold",
        )
        ann2 = self.ax.annotate(
            f"{last_mean:.1f}",
            xy=(last_ep, last_mean),
            xytext=(6, -12), textcoords="offset points",
            color=_CLR_MEAN, fontsize=8, fontweight="bold",
        )
        self._annotations = [ann1, ann2]

    # ──────────────────────────────────────────────────────────────────────────
    def save(self, path: str):
        """
        Export the current figure to a PNG file.

        Parameters
        ----------
        path : output file path (e.g. "training_curve.png")
        """
        self.fig.savefig(path, dpi=150, bbox_inches="tight",
                         facecolor=self.fig.get_facecolor())
        print(f"[Plot] Training curve saved → {path}")


# ─── Module-level singleton & convenience wrapper ────────────────────────────

_live_plot: LivePlot | None = None


def plot(scores: list, mean_scores: list,
         filename: str | None = None,
         save_only: bool = False):
    """
    Convenience wrapper used by training/train.py.

    Parameters
    ----------
    scores      : per-episode score list
    mean_scores : rolling-mean score list
    filename    : if given, save the figure to this path
    save_only   : if True, skip the live-update step (used for final export)
    """
    global _live_plot

    if not save_only:
        # Lazy-init the singleton on first call
        if _live_plot is None:
            try:
                _live_plot = LivePlot()
            except Exception as exc:
                # Headless / no display — degrade gracefully
                print(f"[Plot] Warning: could not open live plot ({exc}). "
                      "Scores will still be saved at the end.")
                _live_plot = None   # keep None so we retry next episode

        if _live_plot is not None:
            _live_plot.update(scores, mean_scores)

    # Export final figure
    if filename and scores:
        if _live_plot is not None:
            _live_plot.save(filename)
        else:
            # Fallback: create a static figure and save it
            _save_static(scores, mean_scores, filename)


# ─── Fallback static save (headless environments) ────────────────────────────

def _save_static(scores: list, mean_scores: list, path: str):
    """Create and save a static plot (used when no display is available)."""
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor(_BG_OUTER)
    ax.set_facecolor(_BG_AXES)

    episodes = list(range(1, len(scores) + 1))
    ax.plot(episodes, scores,      color=_CLR_SCORE, linewidth=1.5,
            alpha=0.75, label="Score / episode")
    ax.plot(episodes, mean_scores, color=_CLR_MEAN,  linewidth=2.5,
            label="Mean score (last 100)")

    for spine in ax.spines.values():
        spine.set_edgecolor(_CLR_SPINE)
    ax.tick_params(colors=_CLR_TEXT)
    ax.set_xlabel("Episode", color=_CLR_TEXT)
    ax.set_ylabel("Score",   color=_CLR_TEXT)
    ax.set_title("NeuroSnake — DQN Training Progress",
                 color=_CLR_TEXT, fontweight="bold")
    ax.grid(True, color=_CLR_GRID, linestyle="--", linewidth=0.6, alpha=0.8)
    ax.legend(facecolor=_BG_AXES, edgecolor=_CLR_SPINE, labelcolor=_CLR_TEXT)

    fig.savefig(path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"[Plot] Training curve saved (static) → {path}")
