import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.pipeline import Pipeline

from data_utils import load_master


def get_rf(model):
    """
    From a loaded model (Pipeline or plain estimator), extract the RandomForest
    that has feature_importances_.
    """
    # If it's already an RF with feature_importances_
    if hasattr(model, "feature_importances_"):
        return model

    # If it's a Pipeline, take the last step that has feature_importances_
    if isinstance(model, Pipeline):
        last_est = model.steps[-1][1]
        if hasattr(last_est, "feature_importances_"):
            return last_est
        else:
            raise ValueError(
                "Pipeline last step has no feature_importances_. "
                f"Steps: {[name for name, _ in model.steps]}"
            )

    raise ValueError(
        "Model is neither a RandomForest nor a Pipeline with RandomForest at the end."
    )


def main():
    # Ensure output folders exist
    os.makedirs("results/figures", exist_ok=True)
    os.makedirs("results/tables", exist_ok=True)

    # We don't actually need df here except to load for consistency
    _ = load_master("data/AlAin_Stations_2007_2025_clean.csv")

    # Load the trained AQI model (pipeline)
    model_aqi = joblib.load("models/rf_aqi.joblib")
    rf_AQI = get_rf(model_aqi)

    importances = rf_AQI.feature_importances_
    n_imp = len(importances)
    print(f"[rf_importance] Number of importances (internal features): {n_imp}")

    # Since the pipeline created internal features, we don't have a 1:1 mapping
    # to the original column names. We use generic names.
    feat_names = [f"feature_{i+1}" for i in range(n_imp)]

    # Build DataFrame
    imp_df = (
        pd.DataFrame({"feature": feat_names, "importance": importances})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )

    # Save table
    out_csv = "results/tables/rf_AQI_feature_importance.csv"
    imp_df.to_csv(out_csv, index=False)
    print(f"Saved RF AQI feature importance to: {out_csv}")

    # Plot top 10
    top_n = 10
    imp_top = imp_df.head(top_n)

    plt.figure(figsize=(7, 5))
    plt.barh(imp_top["feature"][::-1], imp_top["importance"][::-1])
    plt.xlabel("Feature importance (Gini)")
    plt.title("Random Forest AQI model – top internal features")
    plt.tight_layout()

    out_fig = "results/figures/rf_AQI_feature_importance.png"
    plt.savefig(out_fig, dpi=300)
    plt.close()
    print(f"Saved RF AQI feature importance plot to: {out_fig}")


if __name__ == "__main__":
    main()

