"""
Inference script for the enhanced stacking model.
Usage:
    python inference.py --input your_data.csv --output predictions.csv
    
The input should be the concat_features (FC + sMRI + phenotypes) that the pipeline produces.
For new subjects, you need to:
1. Extract FC features using the same method as fc_agent
2. Extract sMRI features using the same method as smri_agent
3. Extract phenotype features using the same method as phenotype_agent
4. Concatenate them in the same order as fusion_agent does
"""

import argparse
import numpy as np
import joblib
import pandas as pd

def load_model(model_path="models/enhanced_stacking_model.joblib"):
    """Load the saved stacking model."""
    model_data = joblib.load(model_path)
    return model_data

def predict(model_data, X_raw):
    """
    Make predictions on raw concat features.
    
    Args:
        model_data: Loaded model data from joblib
        X_raw: Raw concat features array (n_samples, n_features)
    
    Returns:
        predictions: Binary predictions (0=HC, 1=ASD)
        probabilities: Probability scores for ASD class
    """
    X_arr = np.nan_to_num(np.array(X_raw, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    X_scaled = model_data["scaler"].transform(X_arr)
    X_sel = X_scaled[:, model_data["top_idx"]]
    
    X_s = model_data["selector"].transform(X_sel)
    
    P1 = np.zeros((X_sel.shape[0], 3), dtype=np.float32)
    P1[:, 0] = model_data["xgb_model"].predict_proba(X_s)[:, 1]
    P1[:, 1] = model_data["lgbm_model"].predict_proba(X_s)[:, 1]
    P1[:, 2] = model_data["lr_model"].predict_proba(X_s)[:, 1]
    
    P2 = np.zeros((X_sel.shape[0], 3), dtype=np.float32)
    P2[:, 0] = model_data["xgb_meta"].predict_proba(P1)[:, 1]
    P2[:, 1] = model_data["lgbm_meta"].predict_proba(P1)[:, 1]
    P2[:, 2] = model_data["lr_meta"].predict_proba(P1)[:, 1]
    
    interactions = np.column_stack([
        P2[:, 0] * P2[:, 1],
        P2[:, 0] * P2[:, 2],
        P2[:, 1] * P2[:, 2],
        (P2[:, 0] + P2[:, 1]) / 2,
        np.abs(P2[:, 0] - P2[:, 1]),
    ])
    
    P3 = np.zeros(X_sel.shape[0], dtype=np.float32)
    for model in model_data["level3_models"]:
        P3 += model.predict_proba(interactions)[:, 1]
    P3 /= len(model_data["level3_models"])
    
    probs = (P1.mean(axis=1) + P2.mean(axis=1) + P3) / 3.0
    predictions = (probs >= model_data["threshold"]).astype(int)
    
    return predictions, probs

def main():
    parser = argparse.ArgumentParser(description="Run inference with the stacking model")
    parser.add_argument("--model", default="models/enhanced_stacking_model.joblib", help="Path to saved model")
    parser.add_argument("--input", required=True, help="Path to input CSV file with concat features")
    parser.add_argument("--output", default="predictions.csv", help="Path to output predictions CSV")
    
    args = parser.parse_args()
    
    print(f"Loading model from {args.model}...")
    model_data = load_model(args.model)
    
    print(f"Loading data from {args.input}...")
    df = pd.read_csv(args.input)
    
    # Extract features (assuming first column is subject ID)
    if "SUB_ID" in df.columns:
        subject_ids = df["SUB_ID"].values
        X = df.drop(columns=["SUB_ID"]).values
    else:
        subject_ids = [f"Subject_{i}" for i in range(len(df))]
        X = df.values
    
    print(f"Running inference on {X.shape[0]} subjects...")
    predictions, probabilities = predict(model_data, X)
    
    # Create results dataframe
    results = pd.DataFrame({
        "SUB_ID": subject_ids,
        "Prediction": predictions,
        "Probability_ASD": probabilities,
        "Diagnosis": ["ASD" if p == 1 else "HC" for p in predictions]
    })
    
    results.to_csv(args.output, index=False)
    print(f"Predictions saved to {args.output}")
    print(f"\nSummary:")
    print(f"  Total subjects: {len(results)}")
    print(f"  ASD predicted: {sum(predictions)}")
    print(f"  HC predicted: {len(predictions) - sum(predictions)}")
    print(f"  Threshold used: {model_data['threshold']:.4f}")

if __name__ == "__main__":
    main()
