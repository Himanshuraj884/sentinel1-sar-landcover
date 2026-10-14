"""Run the Sentinel-1 land-cover pipeline."""
import argparse
import json
import warnings
from pathlib import Path

import numpy as np

from src.data import load_folder, make_synthetic, split_indices
from src.eda import run_eda
from src.sar import extract_features, preprocess, to_cnn_input
from src.train_rf import run_rf


def results_markdown(s):
    rf, lines = s["random_forest"], []
    lines.append(f"Split: **{s['split']}** | patches: {s['n_patches']} (train {s['n_train']} / val {s['n_val']} / test {s['n_test']})"
                 + (" | **SYNTHETIC DATA, NOT RESULTS**" if s["synthetic"] else ""))
    lines += ["", "| Model | Test accuracy | Macro-F1 | CV accuracy |", "|---|---|---|---|",
              f"| Random Forest (13 SAR features) | {rf['test_accuracy']:.3f} | {rf['test_macro_f1']:.3f} | "
              f"{rf['cv_accuracy_mean']:.3f} +/- {rf['cv_accuracy_std']:.3f} ({rf['cv_kind']}) |"]
    if "cnn" in s:
        c = s["cnn"]
        lines.append(f"| Keras CNN (VV, VH, ratio) | {c['test_accuracy']:.3f} | {c['test_macro_f1']:.3f} | - |")
    lines += ["", "Polarisation ablation (RF test accuracy):", "", "| Features | Accuracy |", "|---|---|"]
    lines += [f"| {k} | {v:.3f} |" for k, v in rf["ablation_test_accuracy"].items()]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/sar")
    ap.add_argument("--synthetic", action="store_true", help="synthetic data, smoke test only")
    ap.add_argument("--split", choices=["group", "random"], default="group")
    ap.add_argument("--input-db", action="store_true", help="input is already in dB")
    ap.add_argument("--max-per-class", type=int, default=0, help="0 = all")
    ap.add_argument("--no-despeckle", action="store_true")
    ap.add_argument("--skip-cnn", action="store_true")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    out = Path(a.out)
    (out / "reports").mkdir(parents=True, exist_ok=True)

    if a.synthetic:
        X, y, groups, names = make_synthetic(seed=a.seed)
    else:
        X, y, groups, names = load_folder(a.data_dir, a.input_db, a.max_per_class, a.seed)
    print(f"Loaded {len(y)} patches, {len(names)} classes: {names}")

    X_db = preprocess(X, despeckle=not a.no_despeckle)
    F = extract_features(X_db)
    tr, va, te = split_indices(y, groups, a.split, a.seed)

    summary = {"synthetic": bool(a.synthetic), "split": a.split, "despeckle": not a.no_despeckle,
               "n_patches": int(len(y)), "n_train": int(len(tr)), "n_val": int(len(va)), "n_test": int(len(te)),
               "classes": names}
    summary["eda"] = run_eda(F, y, names, out / "figures", a.seed)
    summary["random_forest"] = run_rf(F, y, tr, va, te, groups, names, out, a.split, a.seed)

    if not a.skip_cnn:
        try:
            from src.train_cnn import run_cnn
            summary["cnn"] = run_cnn(to_cnn_input(X_db), y, tr, va, te, names, out, a.epochs, seed=a.seed)
        except ImportError:
            warnings.warn("TensorFlow not installed: skipping CNN (use --skip-cnn to silence).")

    (out / "reports" / "summary.json").write_text(json.dumps(summary, indent=2))
    md = results_markdown(summary)
    (out / "reports" / "results.md").write_text(md)
    print("\n" + md)


if __name__ == "__main__":
    main()
