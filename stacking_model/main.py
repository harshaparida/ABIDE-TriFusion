import os
import argparse
from src.agentic.pipeline import run_pipeline, save_report


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, default=os.path.join(os.path.dirname(__file__), "data"))
    p.add_argument("--out", type=str, default="results.json")
    p.add_argument("--quiet", action="store_true")
    p.add_argument("--tune-trials", type=int, default=0, help="Run Optuna tuning with this many trials")
    args = p.parse_args()
    verbose = not args.quiet
    if verbose:
        print("Starting ASD vs HC pipeline", flush=True)
    if args.tune_trials and args.tune_trials > 0:
        # run graph up to fuse, then optuna tune
        from src.agentic.pipeline import (
            load_phenotypes, load_smri_table, load_fc_matrices, align_subjects,
            fc_feature_agent, smri_agent, phenotype_agent, harmonization_agent, fusion_agent,
            optuna_tune_agent
        )
        data_dir = args.data_dir
        if verbose:
            print("Pipeline: loading data", flush=True)
        fc_dir = os.path.join(data_dir, "fc_matrices")
        ph_fp = os.path.join(data_dir, "phenotypic_data.csv")
        smri_fp_1 = os.path.join(data_dir, "structural_data_cleaned.csv")
        if os.path.exists(smri_fp_1):
            smri = load_smri_table(smri_fp_1)
        else:
            smri = None
        ph = load_phenotypes(ph_fp)
        fc_dict = load_fc_matrices(fc_dir, verbose=verbose)
        if smri is None:
            import pandas as pd
            smri = pd.DataFrame({"SUB_ID": []})
        sids, fc_f, smri_f, ph_f = align_subjects(fc_dict, smri, ph)
        init_state = {"fc_dict": fc_f, "smri": smri_f, "phenotypes": ph_f, "subject_ids": sids, "verbose": verbose}
        # run agents sequentially up to fusion
        s1 = {**init_state}
        s2 = {**s1, **fc_feature_agent(s1)}
        s3 = {**s2, **smri_agent(s2)}
        s4 = {**s3, **phenotype_agent(s3)}
        s5 = {**s4, **harmonization_agent(s4)}
        s6 = {**s5, **fusion_agent(s5)}
        state_for_tune = {**s6, "tune_trials": args.tune_trials, "verbose": verbose}
        res = optuna_tune_agent(state_for_tune)
    else:
        res = run_pipeline(args.data_dir, verbose=verbose)
    save_report(res, args.out)
    mr = res.get("model_results", {})
    best = res.get("best_model")
    if mr:
        print("\n=== Final Model Results ===")
        for k, v in sorted(mr.items(), key=lambda x: x[1].get("accuracy", -1) if isinstance(x[1], dict) else -1, reverse=True):
            if isinstance(v, dict):
                print(f"  {k:45s}  acc={v.get('accuracy', float('nan')):.4f}  f1={v.get('f1', float('nan')):.4f}  auc={v.get('auc', float('nan')):.4f}")
    else:
        print("model_results: (empty)")
    print(f"\nbest_model: {best}")
    if best and best in mr:
        bv = mr[best]
        print(f"  accuracy : {bv.get('accuracy', float('nan')):.4f}")
        print(f"  f1       : {bv.get('f1', float('nan')):.4f}")
        print(f"  auc      : {bv.get('auc', float('nan')):.4f}")


if __name__ == "__main__":
    main()
