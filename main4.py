import os
import argparse
from src.agentic.pipeline4 import run_pipeline, save_report


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, default=os.path.join(os.path.dirname(__file__), "data"))
    p.add_argument("--out", type=str, default="resultspipeline4.json")
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args()
    verbose = not args.quiet
    
    if verbose:
        print("=" * 80, flush=True)
        print("ASD vs HC Pipeline (v4): 10-fold CV, Improved Models, Best Single + Majority Vote", flush=True)
        print("=" * 80, flush=True)
    
    res = run_pipeline(args.data_dir, verbose=verbose)
    save_report(res, args.out)
    
    mr = res.get("model_results", {})
    best = res.get("best_model")
    if mr:
        print("\n=== Final Model Results ===", flush=True)
        for k, v in sorted(mr.items(), key=lambda x: x[1].get("accuracy", -1) if isinstance(x[1], dict) else -1, reverse=True):
            if isinstance(v, dict):
                print(f"  {k:45s}  acc={v.get('accuracy', float('nan')):.4f}  f1={v.get('f1', float('nan')):.4f}  auc={v.get('auc', float('nan')):.4f}", flush=True)
    else:
        print("model_results: (empty)", flush=True)
    print(f"\nbest_model: {best}", flush=True)
    if best and best in mr:
        bv = mr[best]
        print(f"  accuracy : {bv.get('accuracy', float('nan')):.4f}", flush=True)
        print(f"  f1       : {bv.get('f1', float('nan')):.4f}", flush=True)
        print(f"  auc      : {bv.get('auc', float('nan')):.4f}", flush=True)


if __name__ == "__main__":
    main()
