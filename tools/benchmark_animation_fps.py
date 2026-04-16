"""Benchmark napari slider animation FPS.

Drives ``QtDims.play`` across a sweep of requested FPS values and records both
the *imposed* (timer tick) and *effective* (frame actually drawn, after the
``_play_ready`` debounce) intervals. Writes a CSV of summary stats and a PNG
scatter of imposed/effective vs requested FPS.

Requires the instrumentation added to ``AnimationThread`` (``_imposed_log``,
``_effective_log``) and ``QtDims._set_frame``. Run from the repo root:

    python tools/benchmark_animation_fps.py --out out/fps_win

Use ``--help`` for options. The viewer window will briefly appear; don't
interact with it.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from qtpy.QtCore import QEventLoop, QTimer

import napari


def _pump(ms: int) -> None:
    """Block for ``ms`` milliseconds while pumping the Qt event loop."""
    loop = QEventLoop()
    QTimer.singleShot(ms, loop.quit)
    loop.exec() if hasattr(loop, 'exec') else loop.exec_()


def run_one(dims_widget, fps: float, duration: float, settle_ms: int = 300):
    """Play ``axis=0`` at ``fps`` for ``duration`` seconds; return logs."""
    thread = dims_widget._animation_thread
    thread._imposed_log = []
    thread._effective_log = []

    dims_widget.play(axis=0, fps=fps)
    _pump(int(duration * 1000))
    dims_widget.stop()
    _pump(settle_ms)  # let the thread finish + pending frames drain

    imposed = list(thread._imposed_log)
    effective = list(thread._effective_log)
    thread._imposed_log = None
    thread._effective_log = None
    return imposed, effective


def summarize(label: str, intervals: list[tuple[float, float]],
              warmup: int = 2) -> dict:
    """Drop first ``warmup`` samples, then compute FPS stats from intervals."""
    samples = [iv for _, iv in intervals[warmup:] if iv > 0]
    if not samples:
        return {
            f'{label}_n': 0,
            f'{label}_mean': float('nan'),
            f'{label}_median': float('nan'),
            f'{label}_p10': float('nan'),
            f'{label}_p90': float('nan'),
        }
    fps = np.array([1.0 / s for s in samples])
    return {
        f'{label}_n': len(fps),
        f'{label}_mean': float(fps.mean()),
        f'{label}_median': float(np.median(fps)),
        f'{label}_p10': float(np.percentile(fps, 10)),
        f'{label}_p90': float(np.percentile(fps, 90)),
    }


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_raw(fps_req: float, imposed: list, effective: list, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['kind', 'requested_fps', 'interval_s'])
        for req, iv in imposed:
            writer.writerow(['imposed', req, iv])
        for req, iv in effective:
            writer.writerow(['effective', req, iv])


def plot(rows: list[dict], path: Path) -> None:
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    req = np.array([r['requested_fps'] for r in rows], dtype=float)
    imp_med = np.array([r['imposed_median'] for r in rows], dtype=float)
    eff_med = np.array([r['effective_median'] for r in rows], dtype=float)
    imp_p10 = np.array([r['imposed_p10'] for r in rows], dtype=float)
    imp_p90 = np.array([r['imposed_p90'] for r in rows], dtype=float)
    eff_p10 = np.array([r['effective_p10'] for r in rows], dtype=float)
    eff_p90 = np.array([r['effective_p90'] for r in rows], dtype=float)

    fig, (ax, ax_err) = plt.subplots(1, 2, figsize=(14, 7))

    ax.fill_between(req, imp_p10, imp_p90, alpha=0.15, color='C0')
    ax.fill_between(req, eff_p10, eff_p90, alpha=0.15, color='C1')
    ax.plot(req, imp_med, 'o-', color='C0', label='imposed (timer tick)')
    ax.plot(req, eff_med, 's-', color='C1', label='effective (drawn)')

    lim_lo = max(min(req.min(), 1.0) * 0.5, 0.1)
    lim_hi = max(req.max() * 1.5, imp_med.max() * 1.2 if imp_med.size else 1.0)
    ax.plot([lim_lo, lim_hi], [lim_lo, lim_hi], 'k--', alpha=0.3, label='y = x')

    ax.set_xlabel('Requested FPS')
    ax.set_ylabel('Measured FPS (median, shaded = 10–90th pct)')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(lim_lo, lim_hi)
    ax.set_ylim(lim_lo, lim_hi)
    ax.grid(True, which='both', alpha=0.3)
    ax.legend(loc='upper left')
    ax.set_title('Imposed vs effective FPS')

    err_req = req - eff_med
    err_imp = imp_med - eff_med
    ax_err.axhline(0, color='k', alpha=0.3, linestyle='--')
    ax_err.plot(req, err_req, 'o-', color='C2',
                label='requested - effective')
    ax_err.plot(req, err_imp, 's-', color='C3',
                label='imposed - effective')
    ax_err.set_xlabel('Requested FPS')
    ax_err.set_ylabel('Error (FPS)')
    ax_err.set_xscale('log')
    ax_err.grid(True, which='both', alpha=0.3)
    ax_err.legend(loc='upper left')
    ax_err.set_title('FPS error vs requested')

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        '--fps',
        type=float,
        nargs='+',
        default=[1, 2, 5, 10, 20, 30, 60, 90, 120, 180, 240, 360, 480],
        help='Requested FPS values to sweep.',
    )
    p.add_argument(
        '--duration',
        type=float,
        default=5.0,
        help='Seconds of playback per FPS step.',
    )
    p.add_argument(
        '--frames',
        type=int,
        default=500,
        help='Number of frames along the animated axis (must exceed fps*duration).',
    )
    p.add_argument(
        '--shape',
        type=int,
        nargs=2,
        default=[512, 512],
        help='YX shape of each frame.',
    )
    p.add_argument(
        '--out',
        type=Path,
        default=Path('animation_fps_benchmark'),
        help='Output path prefix (without extension). Writes <prefix>.csv and <prefix>.png.',
    )
    p.add_argument(
        '--save-raw',
        action='store_true',
        help='Also write per-tick raw intervals to <prefix>_raw_<fps>.csv.',
    )
    p.add_argument(
        '--warmup',
        type=int,
        default=2,
        help='Number of initial samples to drop per FPS step.',
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    max_needed = int(max(args.fps) * args.duration) + 10
    if args.frames < max_needed:
        print(
            f'warning: --frames={args.frames} may be insufficient for the '
            f'highest requested fps over {args.duration}s (need ~{max_needed}); '
            f'loop wraps will still work but may skew timing.',
            file=sys.stderr,
        )

    data = np.random.randint(
        0, 255, size=(args.frames, *args.shape), dtype=np.uint8
    )
    viewer = napari.Viewer(show=True)
    viewer.add_image(data, name='benchmark')
    dims_widget = viewer.window._qt_viewer.dims

    _pump(500)  # let the first frame render

    rows: list[dict] = []
    for fps in args.fps:
        print(f'[benchmark] requested_fps={fps}')
        imposed, effective = run_one(dims_widget, float(fps), args.duration)
        row: dict = {'requested_fps': float(fps)}
        row.update(summarize('imposed', imposed, args.warmup))
        row.update(summarize('effective', effective, args.warmup))
        rows.append(row)
        if args.save_raw:
            write_raw(
                float(fps),
                imposed,
                effective,
                args.out.with_name(f'{args.out.name}_raw_{fps}.csv'),
            )
        print(
            f'  imposed  median={row["imposed_median"]:.2f} fps '
            f'(n={row["imposed_n"]})'
        )
        print(
            f'  effective median={row["effective_median"]:.2f} fps '
            f'(n={row["effective_n"]})'
        )

    csv_path = args.out.with_suffix('.csv')
    png_path = args.out.with_suffix('.png')
    write_csv(rows, csv_path)
    plot(rows, png_path)
    print(f'[benchmark] csv -> {csv_path}')
    print(f'[benchmark] plot -> {png_path}')

    viewer.close()
    return 0


if __name__ == '__main__':
    sys.exit(main())
