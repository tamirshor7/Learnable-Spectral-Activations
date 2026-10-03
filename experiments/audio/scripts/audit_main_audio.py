from pathlib import Path
import json
import math
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

def read_values(group):
    values = []
    root = ROOT / "outputs" / "runs" / group
    for p in sorted(root.rglob("summary.json")):
        try:
            d = json.loads(p.read_text())
        except Exception:
            continue
        for k in ["psnr", "best_psnr", "val_psnr", "best_psnr_floor_1e12"]:
            if k in d:
                try:
                    x = float(d[k])
                except Exception:
                    x = float("nan")
                if math.isfinite(x):
                    values.append(x)
                    break
    if values:
        return values
    p = ROOT / "outputs" / "summaries" / f"{group}.csv"
    if p.exists():
        df = pd.read_csv(p)
        for k in ["psnr", "best_psnr", "val_psnr", "best_psnr_floor_1e12"]:
            if k in df.columns:
                return [float(x) for x in pd.to_numeric(df[k], errors="coerce").dropna()]
    return values

expected = pd.read_csv(ROOT / "results" / "expected" / "main_paper_audio_expected.csv")
rows = []
for _, r in expected.iterrows():
    vals = read_values(str(r["group"]))
    s = pd.Series(vals, dtype=float)
    mean = float(s.mean()) if len(s) else float("nan")
    rows.append({
        "section": r["section"],
        "dataset": r["dataset"],
        "method": r["method"],
        "config": r["config"],
        "group": r["group"],
        "n": int(len(vals)),
        "target_n": int(r["target_n"]),
        "complete": int(len(vals)) == int(r["target_n"]),
        "mean_psnr": mean,
        "expected_mean_psnr": float(r["expected_mean_psnr"]),
        "diff_vs_expected": mean - float(r["expected_mean_psnr"]) if len(vals) else float("nan"),
    })

out = pd.DataFrame(rows)
rep = ROOT / "results" / "reproduced"
rep.mkdir(parents=True, exist_ok=True)
out.to_csv(rep / "main_paper_audio_recreated_summary.csv", index=False)
bad = out[(~out["complete"].astype(bool)) | (out["diff_vs_expected"].abs().fillna(999) > 0.01)]
lines = ["MAIN_PAPER_AUDIO_RECREATION", out.to_string(index=False), "", "MAIN_PAPER_AUDIO_STATUS=" + ("PASS" if len(bad) == 0 else "FAIL")]
if len(bad):
    lines += ["", bad.to_string(index=False)]
(rep / "main_paper_audio_audit.txt").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
if len(bad):
    raise SystemExit(1)
