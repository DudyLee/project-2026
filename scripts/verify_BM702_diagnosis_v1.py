"""Read-only verification of diagnosis tables, algebra and interpretation inputs."""
from pathlib import Path
import json
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pvsd_trial as api

OUT = ROOT/"reports/diagnosis_BM702_v1"


def main():
    run = json.loads((OUT/"run_manifest.json").read_text(encoding="utf-8"))
    for relative, digest in run["input_hashes"].items():
        assert api.sha256(ROOT/relative) == digest, relative
    room = pd.read_csv(OUT/"native_room_metrics.csv")
    focal = room[room.room == "BM702"].set_index("target")
    original = pd.read_csv(ROOT/"reports/holdout_trial_v0_1/room_metrics.csv")
    original = original[(original.room == "BM702")&(original.method == "Frozen_trial_v0_1")].set_index("target")
    for t in focal.index:
        for c in ["mae", "rmse", "r2", "bias"]:
            np.testing.assert_allclose(focal.loc[t,c], original.loc[t,c], atol=1e-10)
    strata = pd.read_csv(OUT/"BM702_stratified_errors.csv")
    count = 0
    for (target, axis), g in strata.groupby(["target", "axis"]):
        assert g.n.sum() == 17520
        np.testing.assert_allclose(np.average(g.mae, weights=g.n), focal.loc[target,"mae"], atol=1e-10)
        np.testing.assert_allclose(np.average(g.bias, weights=g.n), focal.loc[target,"bias"], atol=1e-10)
        np.testing.assert_allclose(g.fraction_of_target_absolute_error.sum(), 1, atol=1e-12)
        count += 1
    q = pd.read_parquet(OUT/"P10_components.parquet")
    np.testing.assert_allclose(q.mean_term+q.ratio_term+q.interaction_term, q.pred_P10-q.true_P10, atol=1e-9)
    np.testing.assert_allclose(q.ratio_true*(q.true_mean+1), q.true_P10, atol=1e-10)
    oracles = pd.read_csv(OUT/"P10_oracle_diagnostics.csv").set_index("diagnostic")
    predictions = {"native_pair":q.pred_P10, "ORACLE_true_mean_only":q.ratio_pred*(q.true_mean+1),
                   "ORACLE_true_ratio_only":q.ratio_true*(q.paired_mean+1), "ORACLE_both":q.ratio_true*(q.true_mean+1)}
    for key, p in predictions.items():
        independent = [mean_absolute_error(q.true_P10,p), np.sqrt(mean_squared_error(q.true_P10,p)), r2_score(q.true_P10,p)]
        np.testing.assert_allclose(oracles.loc[key,["mae","rmse","r2"]].to_numpy(dtype=float), independent, atol=1e-10)
    a = pd.read_csv(OUT/"RF_ratio_room_attribution.csv")
    np.testing.assert_allclose(a.mean_leaf_weight.sum(),1,atol=1e-12)
    np.testing.assert_allclose(a.mean_ratio_contribution.sum(),q.ratio_pred.mean(),atol=1e-12)
    probe = pd.read_csv(OUT/"orientation_branch_probe.csv")
    assert (probe.maximum_absolute_prediction_change < 1e-10).all()
    assert pd.read_csv(OUT/"data_checks.csv").violations.eq(0).all()
    assert pd.read_csv(OUT/"daylight_consistency_checks.csv").drop(columns=["room","Sensor_Count"]).eq(0).all().all()
    verdict = {"status":"passed", "native_scores_match_frozen_holdout":True,"stratified_partition_checks":count,
               "independent_P10_oracle_score_checks":4,"P10_identity_checked":True,"RF_attribution_weight_and_prediction_sums_checked":True,
               "source_and_model_hashes_unchanged":True,"no_estimator_fits":True,
               "limits":"Internal data and frozen-model mechanism verification, not a check against unavailable source simulation grids."}
    (OUT/"verification.json").write_text(json.dumps(verdict,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(verdict,ensure_ascii=False,indent=2))


if __name__ == "__main__":
    main()
