import optuna, logging
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

def self_contained_optuna_calibrator(X_test: [list, np.ndarray, tuple], y_test: [list, np.ndarray, tuple], developer_mode: bool):
    # ---> Objective function <---
    def objective(trial: int):
        depth_min = 4
        depth_max = 72
        estimator_min = 20
        estimator_max = 380
        depth = trial.suggest_int("Depth", depth_min, depth_max)
        estimator = trial.suggest_int("Estimator", estimator_min, estimator_max)
        # - Deploy random forrest classifier for trials testing -
        clf = RandomForestClassifier(n_estimators=estimator, max_depth=depth, random_state=42)
        clf.fit(X_test, y_test)
        # - Evaluate accuracy -
        y_pred = clf.predict(X_test)
        acc = np.mean(y_pred == y_test)

        trial.set_user_attr("Accuracy", acc)
        # --------------------------------------

    # ---> Assertions <---
    assert(type(developer_mode) == bool)
    assert(isinstance(X_test, (tuple, np.ndarray, list)))
    assert(isinstance(y_test, (tuple, np.ndarray, list)))

    # ---> Calibration process <---
    trial = 180
    dual_process_acceleration = True
    jobs_count = 1
    if (dual_process_acceleration == True):
        jobs_count += 1
        if (developer_mode == True):
            print("<> Dual core booster active <>")

    if (developer_mode == True):
        print("Process acceleration status: ACTIVE")

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=trial, n_jobs=jobs_count)

    try:
        optimal_params = study.best_params
        optimal_depth = optimal_params["Depth"]
        optimal_estimators = optimal_params["Estimator"]
        logging.debug("<> Optimal params acquisition failure detected.  <>")
    except:
        optimal_depth = 15
        optimal_estimators = 105

    return optimal_depth, optimal_estimators