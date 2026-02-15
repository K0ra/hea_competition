import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import time, logging, warnings, datetime
from Calibrators import self_contained_optuna_calibrator

# - Behaviors -
diagnostic_mode = False
deploy = True
save_to_csv = False
dev_mode = True
testing_mode = True
warnings.filterwarnings("ignore")

# - Calibration selector -
calibrator_activation = True

# - Processing -
file_path = r"C:\Users\Tomy\PycharmProjects\Medical_classification_ML\Data\rand_hrs_model_merged.parquet"
df = pd.read_parquet(file_path)
if (save_to_csv == True):
    file_name = input("File name for local storage: ")
    df.to_csv(f"{file_name.replace(' ', '_')}_{len(df)}.csv", index=True)

if (diagnostic_mode == True):
    print("\nHeaders")
    print(df.head())
    print("\nInfo")
    print(df.info())

isolated_usage_df = df[["target_DIABE", "target_HIBPE", "target_CANCR", "target_LUNGE",
            "target_HEARTE", "target_STROKE", "target_PSYCHE", "target_ARTHRE", "target_DIABE"]].dropna()
if (diagnostic_mode == True):
    print("\nIsolated features and target merged info:")
    print(isolated_usage_df.info())
    print(isolated_usage_df.describe())
    print(f"Number of nan values:\n{isolated_usage_df.isna()}\n")

if (deploy == True):
    deploy_start_time = datetime.datetime.now()
    # > Features <
    x = isolated_usage_df[["target_DIABE", "target_HIBPE", "target_CANCR", "target_LUNGE",
            "target_HEARTE", "target_STROKE", "target_PSYCHE", "target_ARTHRE"]].to_numpy()
    # > Target labels <
    y = isolated_usage_df["target_DIABE"].to_numpy()
    if (dev_mode == True):
        print("<> X and y variable locked in <>")

    # - Train/test splitting -
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.24, random_state=42)
    if (dev_mode == True):
        print("<> Train test split completed <>")

    # - Calibrate the parameters -
    estimators = 0
    depth = 0
    calibration_start = datetime.datetime.now()
    if (calibrator_activation == True):
        if (dev_mode == True):
            print("Calibration initializing")
        optimal_depth, optimal_estimator = self_contained_optuna_calibrator(X_test, y_test, dev_mode)
        # ----------------------------
        if (type(optimal_depth) != int or type(optimal_estimator) != int):
            print("ERROR!!!!! Failure to calibrate. Resetting to default")
            depth += 10
            estimators += 100
        elif (type(optimal_depth) == int and type(optimal_estimator) == int):
            estimators += optimal_estimator
            depth += optimal_depth
        # ----------------------------
    elif (calibrator_activation == False):
        estimators += 100
        depth += 10
    calibration_end  = datetime.datetime.now() - calibration_start

    # - Fit the classifier -
    classification_fitting = RandomForestClassifier(n_estimators=estimators, max_depth=depth, random_state= 42)
    classification_fitting.fit(X_train, y_train)
    deploy_end_time = datetime.datetime.now() - deploy_start_time
    # - Evaluate -
    print("Training set accuracy:", classification_fitting.score(X_train, y_train))
    print("Test set accuracy:", classification_fitting.score(X_test, y_test))
    print(f"Calibration time: {calibration_end}")
    print(f"End time for deployment: {deploy_end_time}")
elif (deploy == False):
    print("Deployment not cleared. Manually activate the deployment component.")