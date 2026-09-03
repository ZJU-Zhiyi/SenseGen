import argparse
from pathlib import Path
from typing import Iterable

import numpy as np


def _format_bytes(num_bytes: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    value = float(num_bytes)
    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            return f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{value:.2f} TB"


def _safe_stats(arr: np.ndarray):
    if arr.size == 0:
        return {"empty": True}

    if np.iscomplexobj(arr):
        real = arr.real
        imag = arr.imag
        amp = np.abs(arr)
        return {
            "empty": False,
            "real_min": float(np.nanmin(real)),
            "real_max": float(np.nanmax(real)),
            "imag_min": float(np.nanmin(imag)),
            "imag_max": float(np.nanmax(imag)),
            "abs_min": float(np.nanmin(amp)),
            "abs_max": float(np.nanmax(amp)),
            "abs_mean": float(np.nanmean(amp)),
            "nan_count": int(np.isnan(real).sum() + np.isnan(imag).sum()),
            "inf_count": int(np.isinf(real).sum() + np.isinf(imag).sum()),
        }

    return {
        "empty": False,
        "min": float(np.nanmin(arr)),
        "max": float(np.nanmax(arr)),
        "mean": float(np.nanmean(arr)),
        "nan_count": int(np.isnan(arr).sum()) if np.issubdtype(arr.dtype, np.floating) else 0,
        "inf_count": int(np.isinf(arr).sum()) if np.issubdtype(arr.dtype, np.floating) else 0,
    }


def inspect_file(path: Path, show_values: int = 0) -> None:
    try:
        arr = np.load(path, mmap_mode="r")
    except Exception as exc:
        print(f"[ERROR] {path}")
        print(f"  failed to load: {exc}")
        return

    stats = _safe_stats(arr)

    print(f"[NPY] {path}")
    print(f"  shape      : {arr.shape}")
    print(f"  ndim       : {arr.ndim}")
    print(f"  dtype      : {arr.dtype}")
    print(f"  complex    : {np.iscomplexobj(arr)}")
    print(f"  elements   : {arr.size}")
    print(f"  data bytes : {_format_bytes(arr.nbytes)}")

    if stats.get("empty"):
        print("  stats      : empty array")
    elif np.iscomplexobj(arr):
        print(f"  real range : [{stats['real_min']:.6g}, {stats['real_max']:.6g}]")
        print(f"  imag range : [{stats['imag_min']:.6g}, {stats['imag_max']:.6g}]")
        print(f"  abs range  : [{stats['abs_min']:.6g}, {stats['abs_max']:.6g}]")
        print(f"  abs mean   : {stats['abs_mean']:.6g}")
        print(f"  nan / inf  : {stats['nan_count']} / {stats['inf_count']}")
    else:
        print(f"  range      : [{stats['min']:.6g}, {stats['max']:.6g}]")
        print(f"  mean       : {stats['mean']:.6g}")
        print(f"  nan / inf  : {stats['nan_count']} / {stats['inf_count']}")

    if show_values > 0:
        preview = np.asarray(arr).reshape(-1)[:show_values]
        print(f"  first {len(preview)} values: {preview}")


def iter_npy_files(path: Path, recursive: bool) -> Iterable[Path]:
    if path.is_file():
        if path.suffix == ".npy":
            yield path
        return

    pattern = "**/*.npy" if recursive else "*.npy"
    yield from sorted(path.glob(pattern))


def parse_args():
    parser = argparse.ArgumentParser(description="Inspect .npy file shape, dtype, and basic stats.")
    parser.add_argument("--path", type=str, required=True, help="Path to a .npy file or a folder.")
    parser.add_argument("--recursive", action="store_true", help="Scan subfolders when --path is a folder.")
    parser.add_argument("--show_values", type=int, default=0, help="Print first N flattened values.")
    return parser.parse_args()


def main():
    args = parse_args()
    path = Path(args.path)

    if not path.exists():
        raise FileNotFoundError(f"path does not exist: {path}")

    files = list(iter_npy_files(path, recursive=args.recursive))
    if not files:
        raise ValueError(f"no .npy files found: {path}")

    for i, file_path in enumerate(files):
        if i > 0:
            print("")
        inspect_file(file_path, show_values=args.show_values)

    print("")
    print(f"inspected files: {len(files)}")


if __name__ == "__main__":
    main()
