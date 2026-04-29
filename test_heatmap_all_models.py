from __future__ import annotations

import argparse
import re
import time
from dataclasses import dataclass
from pathlib import Path

from ultralytics import YOLO


@dataclass
class RunResult:
    model: str
    baseline_ok: bool
    heatmap_ok: bool
    error: str = ""


def make_model_names(versions: list[str], sizes: list[str], task_suffixes: list[str]) -> list[str]:
    """Build model checkpoint names from versions, sizes, and task suffixes."""
    stems = {"8": "yolov8", "9": "yolov9", "11": "yolo11", "12": "yolo12", "26": "yolo26"}
    default_sizes_by_version = {
        "8": ["n", "s", "m", "l", "x"],
        "9": ["t", "s", "m", "c", "e"],
        "11": ["n", "s", "m", "l", "x"],
        "12": ["n", "s", "m", "l", "x"],
        "26": ["n", "s", "m", "l", "x"],
    }
    names = []
    for version in versions:
        stem = stems[version]
        selected_sizes = default_sizes_by_version[version] if sizes == ["auto"] else sizes
        for size in selected_sizes:
            for suffix in task_suffixes:
                names.append(f"{stem}{size}{suffix}.pt")
    return names


def safe_name(value: str) -> str:
    """Convert a model name into a filesystem-safe folder name."""
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", value)


def test_model(model_name: str, source: str, imgsz: int, conf: float, opacity: float, out_root: Path) -> RunResult:
    """Run baseline + heatmap predictions for one model and collect status."""
    try:
        model = YOLO(model_name)
    except Exception as e:
        return RunResult(model=model_name, baseline_ok=False, heatmap_ok=False, error=f"load failed: {e}")

    model_dir = out_root / safe_name(model_name)
    baseline_ok, heatmap_ok = False, False

    try:
        model.predict(
            source,
            save=True,
            imgsz=imgsz,
            conf=conf,
            project=str(model_dir),
            name="baseline",
            exist_ok=True,
        )
        baseline_ok = True
    except Exception as e:
        return RunResult(model=model_name, baseline_ok=False, heatmap_ok=False, error=f"baseline failed: {e}")

    try:
        model.predict(
            source,
            save=True,
            imgsz=imgsz,
            conf=conf,
            HeatMap=True,
            HMO=opacity,
            project=str(model_dir),
            name="heatmap",
            exist_ok=True,
        )
        heatmap_ok = True
    except Exception as e:
        return RunResult(
            model=model_name,
            baseline_ok=baseline_ok,
            heatmap_ok=False,
            error=f"heatmap failed: {e}",
        )

    return RunResult(model=model_name, baseline_ok=baseline_ok, heatmap_ok=heatmap_ok)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compatibility test for baseline + HeatMap predict across YOLO versions/variants."
    )
    parser.add_argument("--source", default="https://ultralytics.com/images/bus.jpg", help="Image/video source")
    parser.add_argument("--imgsz", type=int, default=320, help="Inference image size")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--hmo", type=float, default=0.8, help="Heatmap opacity")
    parser.add_argument("--out", default="runs/heatmap-compat", help="Output root directory")
    parser.add_argument("--versions", nargs="+", default=["8", "9", "11", "12", "26"], help="YOLO versions to test")
    parser.add_argument(
        "--sizes",
        nargs="+",
        default=["auto"],
        help="Model size variants to test (default: version-specific detect variants)",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=["detect"],
        choices=["detect", "segment", "pose", "obb", "classify"],
        help="Task variants to test",
    )
    args = parser.parse_args()

    suffix_map = {"detect": "", "segment": "-seg", "pose": "-pose", "obb": "-obb", "classify": "-cls"}
    task_suffixes = [suffix_map[t] for t in args.tasks]
    model_names = make_model_names(args.versions, args.sizes, task_suffixes)
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    print(f"Testing {len(model_names)} models...")
    start = time.time()
    results: list[RunResult] = []
    for idx, model_name in enumerate(model_names, start=1):
        print(f"[{idx}/{len(model_names)}] {model_name}")
        result = test_model(model_name, args.source, args.imgsz, args.conf, args.hmo, out_root)
        results.append(result)
        status = "PASS" if (result.baseline_ok and result.heatmap_ok) else "FAIL"
        msg = f"  -> {status}"
        if result.error:
            msg += f" | {result.error}"
        print(msg)

    passed = sum(1 for r in results if r.baseline_ok and r.heatmap_ok)
    failed = len(results) - passed
    print("\n=== Summary ===")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Elapsed: {time.time() - start:.1f}s")
    print(f"Outputs: {out_root.resolve()}")

    if failed:
        print("\nFailed models:")
        for r in results:
            if not (r.baseline_ok and r.heatmap_ok):
                print(f"- {r.model}: {r.error}")


if __name__ == "__main__":
    main()
