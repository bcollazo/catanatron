"""Plot placement training metrics from one or more JSONL logs."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

try:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import PercentFormatter
except ImportError as exc:  # pragma: no cover - import guard for script usage
    raise SystemExit(
        "matplotlib is required to plot training metrics. "
        "Install with: pip install matplotlib"
    ) from exc

try:
    plt.style.use("seaborn-v0_8-whitegrid")
except OSError:  # pragma: no cover - fallback if style missing
    plt.style.use("ggplot")

TITLE_SIZE = 14
LABEL_SIZE = 12
TICK_SIZE = 10
PLOT_SIZE = (10, 6)
DASHBOARD_SIZE = (14, 10)
RANDOM_BASELINE = 0.20


def _set_axis_fonts(ax, percent_y: bool = False):
    ax.title.set_fontsize(TITLE_SIZE)
    ax.xaxis.label.set_size(LABEL_SIZE)
    ax.yaxis.label.set_size(LABEL_SIZE)
    ax.tick_params(axis="both", labelsize=TICK_SIZE)
    if percent_y:
        ax.yaxis.set_major_formatter(PercentFormatter(1.0))


def _series_name(run_label: str, series_name: str, multi_run: bool) -> str:
    if multi_run:
        return f"{run_label} {series_name}"
    return series_name


def _missing_legend_handle(label: str, color: str, linestyle="-"):
    return Line2D(
        [],
        [],
        color=color,
        linestyle=linestyle,
        label=f"{label} (not available)",
    )


def _append_missing_handle(extra_handles: list[Line2D], label: str, color: str, linestyle="-"):
    missing_label = f"{label} (not available)"
    if any(handle.get_label() == missing_label for handle in extra_handles):
        return
    extra_handles.append(_missing_legend_handle(label, color, linestyle=linestyle))


def _field_series(epoch_rows, field: str):
    xs = [row["epoch"] for row in epoch_rows if field in row]
    ys = [row[field] for row in epoch_rows if field in row]
    return xs, ys


def _best_val_marker(epoch_rows):
    if not epoch_rows:
        return None
    rows = [row for row in epoch_rows if "val_loss" in row]
    if not rows:
        return None
    best_row = min(rows, key=lambda row: row["val_loss"])
    return best_row["epoch"], best_row["val_loss"]


def _save_figure(fig, out_dir: str, stem: str, fmt: str, dpi: int):
    out_path = os.path.join(out_dir, f"{stem}.{fmt}")
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    return out_path


def _add_resume_markers(ax, run_meta):
    for marker in run_meta.get("_resume_markers", []):
        ax.axvline(marker, color="0.7", linestyle=":", linewidth=1, alpha=0.7)


def load_metrics(jsonl_path):
    """Return (run_meta, epoch_rows) sorted by epoch number."""

    run_meta = {}
    epochs = []
    resume_markers = []
    pending_resume_marker = None

    with open(jsonl_path, encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            event = row.get("event")
            if event == "run_started":
                if not run_meta:
                    run_meta = dict(row)
                else:
                    start_epoch = row.get("start_epoch")
                    if start_epoch is not None:
                        resume_markers.append(int(start_epoch))
                    else:
                        pending_resume_marker = True
            elif event == "epoch":
                epochs.append(row)
                if pending_resume_marker:
                    resume_markers.append(int(row["epoch"]))
                    pending_resume_marker = None

    epochs.sort(key=lambda row: row["epoch"])
    run_meta["_resume_markers"] = sorted(
        {marker for marker in resume_markers if marker is not None and marker > 1}
    )
    return run_meta, epochs


def make_label(run_meta, path):
    model_type = str(run_meta.get("model_type", "mlp")).upper()
    games = run_meta.get("games_loaded", "?")
    hidden = run_meta.get("hidden_size", "?")
    lr = run_meta.get("lr", "?")
    return f"{model_type} g={games} h={hidden} lr={lr}"


def plot_loss_curves(runs, out_dir, fmt, dpi):
    fig, ax = plt.subplots(figsize=PLOT_SIZE)
    colors = plt.get_cmap("tab10").colors
    multi_run = len(runs) > 1

    for idx, (label, run_meta, epoch_rows) in enumerate(runs):
        color = colors[idx % len(colors)]
        epochs_train, train_loss = _field_series(epoch_rows, "train_loss")
        epochs_val, val_loss = _field_series(epoch_rows, "val_loss")
        if epochs_train:
            ax.plot(
                epochs_train,
                train_loss,
                color=color,
                linestyle="-",
                linewidth=2,
                label=_series_name(label, "train_loss", multi_run),
            )
        if epochs_val:
            ax.plot(
                epochs_val,
                val_loss,
                color=color,
                linestyle="--",
                linewidth=2,
                label=_series_name(label, "val_loss", multi_run),
            )
            best_epoch, best_val = _best_val_marker(epoch_rows)
            ax.axhline(
                best_val,
                color=color,
                linestyle=":",
                linewidth=1.5,
                alpha=0.8,
                label=_series_name(label, "best_val_loss", multi_run),
            )
            ax.scatter(
                [best_epoch],
                [best_val],
                color=color,
                marker="*",
                s=140,
                zorder=5,
            )
        _add_resume_markers(ax, run_meta)

    ax.set_title("Training & Validation Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid(True, alpha=0.3)
    _set_axis_fonts(ax)
    ax.legend()
    _save_figure(fig, out_dir, "loss_curves", fmt, dpi)
    return fig


def plot_accuracy_curves(runs, out_dir, fmt, dpi):
    fig, ax = plt.subplots(figsize=PLOT_SIZE)
    colors = plt.get_cmap("tab10").colors
    multi_run = len(runs) > 1
    extra_handles = []

    for idx, (label, run_meta, epoch_rows) in enumerate(runs):
        color = colors[idx % len(colors)]
        series_specs = [
            ("val_acc", "-", 2.0),
            ("val_winner_acc", "--", 1.8),
            ("val_loser_acc", "-.", 1.8),
            ("val_settlement_acc", (0, (5, 1)), 1.6),
            ("val_road_acc", (0, (1, 1)), 1.6),
            ("train_acc", ":", 1.8),
        ]
        for field, linestyle, linewidth in series_specs:
            xs, ys = _field_series(epoch_rows, field)
            if xs:
                ax.plot(
                    xs,
                    ys,
                    color=color,
                    linestyle=linestyle,
                    linewidth=linewidth,
                    label=_series_name(label, field, multi_run),
                )
            elif field in {"val_settlement_acc", "val_road_acc", "train_acc"}:
                _append_missing_handle(
                    extra_handles,
                    _series_name(label, field, multi_run),
                    color,
                    linestyle=linestyle,
                )
        _add_resume_markers(ax, run_meta)

    ax.axhline(
        RANDOM_BASELINE,
        color="0.3",
        linestyle="--",
        linewidth=1.2,
        label="random baseline",
    )
    ax.set_title("Accuracy Over Training")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.3)
    _set_axis_fonts(ax, percent_y=True)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles + extra_handles, labels + [h.get_label() for h in extra_handles])
    _save_figure(fig, out_dir, "accuracy_curves", fmt, dpi)
    return fig


def plot_overfitting_gap(runs, out_dir, fmt, dpi):
    fig, ax = plt.subplots(figsize=PLOT_SIZE)
    ax2 = ax.twinx()
    colors = plt.get_cmap("tab10").colors
    multi_run = len(runs) > 1
    acc_axis_used = False
    extra_handles = []

    for idx, (label, run_meta, epoch_rows) in enumerate(runs):
        color = colors[idx % len(colors)]
        gap_x = []
        gap_y = []
        for row in epoch_rows:
            if "train_loss" in row and "val_loss" in row:
                gap_x.append(row["epoch"])
                gap_y.append(row["val_loss"] - row["train_loss"])
        if gap_x:
            ax.plot(
                gap_x,
                gap_y,
                color=color,
                linestyle="-",
                linewidth=2,
                label=_series_name(label, "val_loss-train_loss", multi_run),
            )

        acc_gap_x = []
        acc_gap_y = []
        for row in epoch_rows:
            if "train_acc" in row and "val_acc" in row:
                acc_gap_x.append(row["epoch"])
                acc_gap_y.append(row["train_acc"] - row["val_acc"])
        if acc_gap_x:
            acc_axis_used = True
            ax2.plot(
                acc_gap_x,
                acc_gap_y,
                color=color,
                linestyle="--",
                linewidth=1.8,
                alpha=0.8,
                label=_series_name(label, "train_acc-val_acc", multi_run),
            )
        else:
            _append_missing_handle(
                extra_handles,
                _series_name(label, "train_acc-val_acc", multi_run),
                color,
                linestyle="--",
            )
        _add_resume_markers(ax, run_meta)

    ax.axhline(0.0, color="0.3", linestyle=":", linewidth=1.2)
    ax.set_title("Generalization Gap (val_loss − train_loss)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss Gap")
    ax.grid(True, alpha=0.3)
    _set_axis_fonts(ax)
    if acc_axis_used:
        ax2.set_ylabel("Accuracy Gap")
        _set_axis_fonts(ax2, percent_y=True)

    handles1, labels1 = ax.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(
        handles1 + handles2 + extra_handles,
        labels1 + labels2 + [h.get_label() for h in extra_handles],
    )
    _save_figure(fig, out_dir, "overfitting_gap", fmt, dpi)
    return fig


def plot_lr_and_grad_norm(runs, out_dir, fmt, dpi):
    if not any(any("lr" in row for row in epoch_rows) for _, _, epoch_rows in runs):
        print("Skipping lr_grad_norm plot: lr field not available in provided metrics.")
        return None

    fig, ax = plt.subplots(figsize=PLOT_SIZE)
    ax2 = ax.twinx()
    colors = plt.get_cmap("tab10").colors
    multi_run = len(runs) > 1
    lr_values = []
    extra_handles = []

    for idx, (label, run_meta, epoch_rows) in enumerate(runs):
        color = colors[idx % len(colors)]

        lr_x, lr_y = _field_series(epoch_rows, "lr")
        if lr_x:
            lr_values.extend(value for value in lr_y if value > 0)
            ax.plot(
                lr_x,
                lr_y,
                color=color,
                linestyle="-",
                linewidth=2,
                label=_series_name(label, "lr", multi_run),
            )
        else:
            _append_missing_handle(
                extra_handles,
                _series_name(label, "lr", multi_run),
                color,
                linestyle="-",
            )

        mean_x, mean_y = _field_series(epoch_rows, "mean_grad_norm")
        max_x, max_y = _field_series(epoch_rows, "max_grad_norm")
        if mean_x:
            ax2.plot(
                mean_x,
                mean_y,
                color=color,
                linestyle="--",
                linewidth=1.8,
                label=_series_name(label, "mean_grad_norm", multi_run),
            )
        else:
            _append_missing_handle(
                extra_handles,
                _series_name(label, "mean_grad_norm", multi_run),
                color,
                linestyle="--",
            )
        if max_x:
            ax2.fill_between(
                max_x,
                0,
                max_y,
                color=color,
                alpha=0.10,
                label=_series_name(label, "max_grad_norm", multi_run),
            )
        else:
            _append_missing_handle(
                extra_handles,
                _series_name(label, "max_grad_norm", multi_run),
                color,
                linestyle=":",
            )
        _add_resume_markers(ax, run_meta)

    if lr_values:
        min_lr = min(lr_values)
        max_lr = max(lr_values)
        if min_lr > 0 and max_lr / min_lr > 10:
            ax.set_yscale("log")

    ax.set_title("Learning Rate & Gradient Norm")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Learning Rate")
    ax2.set_ylabel("Gradient Norm")
    ax.grid(True, alpha=0.3)
    _set_axis_fonts(ax)
    _set_axis_fonts(ax2)

    handles1, labels1 = ax.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(
        handles1 + handles2 + extra_handles,
        labels1 + labels2 + [h.get_label() for h in extra_handles],
    )
    _save_figure(fig, out_dir, "lr_grad_norm", fmt, dpi)
    return fig


def plot_settlement_vs_road(runs, out_dir, fmt, dpi):
    if not any(
        any("val_settlement_acc" in row for row in epoch_rows)
        for _, _, epoch_rows in runs
    ):
        print(
            "Skipping settlement_vs_road plot: "
            "val_settlement_acc field not available in provided metrics."
        )
        return None

    fig, ax = plt.subplots(figsize=PLOT_SIZE)
    colors = plt.get_cmap("tab10").colors
    multi_run = len(runs) > 1
    extra_handles = []

    for idx, (label, run_meta, epoch_rows) in enumerate(runs):
        color = colors[idx % len(colors)]
        settle_x, settle_y = _field_series(epoch_rows, "val_settlement_acc")
        road_x, road_y = _field_series(epoch_rows, "val_road_acc")

        if settle_x:
            ax.plot(
                settle_x,
                settle_y,
                color=color,
                linestyle="-",
                linewidth=2,
                label=_series_name(label, "val_settlement_acc", multi_run),
            )
        else:
            _append_missing_handle(
                extra_handles,
                _series_name(label, "val_settlement_acc", multi_run),
                color,
                linestyle="-",
            )

        if road_x:
            ax.plot(
                road_x,
                road_y,
                color=color,
                linestyle="--",
                linewidth=2,
                label=_series_name(label, "val_road_acc", multi_run),
            )
        else:
            _append_missing_handle(
                extra_handles,
                _series_name(label, "val_road_acc", multi_run),
                color,
                linestyle="--",
            )
        _add_resume_markers(ax, run_meta)

    ax.axhline(
        RANDOM_BASELINE,
        color="0.3",
        linestyle="--",
        linewidth=1.2,
        label="random baseline",
    )
    ax.set_title("Settlement vs Road Accuracy")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.3)
    _set_axis_fonts(ax, percent_y=True)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles + extra_handles, labels + [h.get_label() for h in extra_handles])
    _save_figure(fig, out_dir, "settlement_vs_road", fmt, dpi)
    return fig


def plot_dashboard(runs, out_dir, fmt, dpi):
    fig, axes = plt.subplots(2, 2, figsize=DASHBOARD_SIZE)
    colors = plt.get_cmap("tab10").colors
    multi_run = len(runs) > 1
    has_settlement_breakdown = any(
        any("val_settlement_acc" in row for row in epoch_rows)
        for _, _, epoch_rows in runs
    )
    has_lr_data = any(
        any("lr" in row for row in epoch_rows)
        for _, _, epoch_rows in runs
    )

    ax_loss = axes[0, 0]
    ax_acc = axes[0, 1]
    ax_gap = axes[1, 0]
    ax_bottom_right = axes[1, 1]

    for idx, (label, run_meta, epoch_rows) in enumerate(runs):
        color = colors[idx % len(colors)]

        train_x, train_y = _field_series(epoch_rows, "train_loss")
        val_x, val_y = _field_series(epoch_rows, "val_loss")
        if train_x:
            ax_loss.plot(
                train_x,
                train_y,
                color=color,
                linestyle="-",
                linewidth=2,
                label=_series_name(label, "train_loss", multi_run),
            )
        if val_x:
            ax_loss.plot(
                val_x,
                val_y,
                color=color,
                linestyle="--",
                linewidth=2,
                label=_series_name(label, "val_loss", multi_run),
            )
        marker = _best_val_marker(epoch_rows)
        if marker is not None:
            ax_loss.scatter([marker[0]], [marker[1]], color=color, marker="*", s=100)

        val_acc_x, val_acc_y = _field_series(epoch_rows, "val_acc")
        if val_acc_x:
            ax_acc.plot(
                val_acc_x,
                val_acc_y,
                color=color,
                linestyle="-",
                linewidth=2,
                label=_series_name(label, "val_acc", multi_run),
            )
        train_acc_x, train_acc_y = _field_series(epoch_rows, "train_acc")
        if train_acc_x:
            ax_acc.plot(
                train_acc_x,
                train_acc_y,
                color=color,
                linestyle=":",
                linewidth=1.8,
                label=_series_name(label, "train_acc", multi_run),
            )

        gap_x = []
        gap_y = []
        for row in epoch_rows:
            if "train_loss" in row and "val_loss" in row:
                gap_x.append(row["epoch"])
                gap_y.append(row["val_loss"] - row["train_loss"])
        if gap_x:
            ax_gap.plot(
                gap_x,
                gap_y,
                color=color,
                linestyle="-",
                linewidth=2,
                label=_series_name(label, "gap", multi_run),
            )

        settle_x, settle_y = _field_series(epoch_rows, "val_settlement_acc")
        road_x, road_y = _field_series(epoch_rows, "val_road_acc")
        lr_x, lr_y = _field_series(epoch_rows, "lr")
        if has_settlement_breakdown:
            if settle_x:
                ax_bottom_right.plot(
                    settle_x,
                    settle_y,
                    color=color,
                    linestyle="-",
                    linewidth=2,
                    label=_series_name(label, "val_settlement_acc", multi_run),
                )
            if road_x:
                ax_bottom_right.plot(
                    road_x,
                    road_y,
                    color=color,
                    linestyle="--",
                    linewidth=2,
                    label=_series_name(label, "val_road_acc", multi_run),
                )
            ax_bottom_right.set_title("Settlement vs Road Accuracy")
            ax_bottom_right.set_ylabel("Accuracy")
            ax_bottom_right.set_ylim(0.0, 1.0)
            _set_axis_fonts(ax_bottom_right, percent_y=True)
        elif has_lr_data:
            if lr_x:
                ax_bottom_right.plot(
                    lr_x,
                    lr_y,
                    color=color,
                    linestyle="-",
                    linewidth=2,
                    label=_series_name(label, "lr", multi_run),
                )
                ax_bottom_right.set_title("Learning Rate")
                ax_bottom_right.set_ylabel("LR")
                _set_axis_fonts(ax_bottom_right)

        _add_resume_markers(ax_loss, run_meta)
        _add_resume_markers(ax_acc, run_meta)
        _add_resume_markers(ax_gap, run_meta)
        _add_resume_markers(ax_bottom_right, run_meta)

    ax_acc.axhline(
        RANDOM_BASELINE,
        color="0.3",
        linestyle="--",
        linewidth=1.0,
    )
    ax_gap.axhline(0.0, color="0.3", linestyle=":", linewidth=1.0)

    ax_loss.set_title("Loss")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Loss")
    ax_loss.grid(True, alpha=0.3)
    _set_axis_fonts(ax_loss)

    ax_acc.set_title("Accuracy")
    ax_acc.set_xlabel("Epoch")
    ax_acc.set_ylabel("Accuracy")
    ax_acc.set_ylim(0.0, 1.0)
    ax_acc.grid(True, alpha=0.3)
    _set_axis_fonts(ax_acc, percent_y=True)

    ax_gap.set_title("Overfitting Gap")
    ax_gap.set_xlabel("Epoch")
    ax_gap.set_ylabel("val_loss - train_loss")
    ax_gap.grid(True, alpha=0.3)
    _set_axis_fonts(ax_gap)

    if not has_settlement_breakdown and not has_lr_data:
        ax_bottom_right.text(
            0.5,
            0.5,
            "No settlement/road or LR data available",
            ha="center",
            va="center",
            fontsize=12,
            transform=ax_bottom_right.transAxes,
        )
        ax_bottom_right.set_title("Auxiliary Signals")
        _set_axis_fonts(ax_bottom_right)

    ax_bottom_right.set_xlabel("Epoch")
    ax_bottom_right.grid(True, alpha=0.3)

    handles = []
    labels = []
    for axis in axes.flat:
        h, l = axis.get_legend_handles_labels()
        for handle, label in zip(h, l):
            if label not in labels:
                handles.append(handle)
                labels.append(label)
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=2, fontsize=10)

    if len(runs) == 1:
        label, run_meta, _ = runs[0]
        fig.suptitle(
            f"{label} | selection={run_meta.get('selection_mode', '?')} | "
            f"val_frac={run_meta.get('val_frac', '?')}",
            fontsize=TITLE_SIZE,
        )
    else:
        fig.suptitle("Placement Training Dashboard", fontsize=TITLE_SIZE)

    fig.tight_layout(rect=(0, 0, 1, 0.95))
    _save_figure(fig, out_dir, "training_dashboard", fmt, dpi)
    return fig


def main():
    parser = argparse.ArgumentParser(description="Plot placement training metrics")
    parser.add_argument("metrics", nargs="+", help="JSONL metrics file(s)")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--format", default="png", choices=["png", "pdf"])
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()

    out_dir = args.out_dir or os.path.dirname(args.metrics[0]) or "."
    os.makedirs(out_dir, exist_ok=True)

    runs = []
    for path in args.metrics:
        meta, epochs = load_metrics(path)
        label = make_label(meta, path)
        runs.append((label, meta, epochs))

    plot_loss_curves(runs, out_dir, args.format, args.dpi)
    plot_accuracy_curves(runs, out_dir, args.format, args.dpi)
    plot_overfitting_gap(runs, out_dir, args.format, args.dpi)
    plot_lr_and_grad_norm(runs, out_dir, args.format, args.dpi)
    plot_settlement_vs_road(runs, out_dir, args.format, args.dpi)
    plot_dashboard(runs, out_dir, args.format, args.dpi)

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
