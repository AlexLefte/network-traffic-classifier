import numpy  as np
import pandas as pd
import os
import pickle
import argparse
import matplotlib.pyplot as plt

from sklearn.ensemble           import RandomForestClassifier as RF
from sklearn.utils              import shuffle
from sklearn.metrics            import confusion_matrix
from utils                      import read_csv, compute_metrics
from sklearn.utils.class_weight import compute_sample_weight
from imblearn.combine import SMOTEENN


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_file", type=str, required=True)
    parser.add_argument("--smote", action="store_true")
    args = parser.parse_args()
    #
    # Datasets
    file_path = args.csv_file
    #
    # Hyperparameters
    estimators = list(range(5, 20, 5))
    min_samples_split = [0.05, 0.1, 0.15, 0.2]
    max_depth = list(range(3, 10))
    min_samples_leaf = [0.05, 0.1]
    max_samples = [0.3, 0.4, 0.5, 0.6, 0.7]
    max_features = ['sqrt', 'log2']
    #
    Nsim = len(estimators)*len(min_samples_split)*len(max_depth)* \
      len(min_samples_leaf)*len(max_samples)*len(max_features)
    idx_sim = 0
    #
    # Save best model for the current split
    best_model = None
    best_val_score = -float('inf')
    #
    # Read and shuffle the data
    class_of_interest = [14, 15, 16, 17]
    # class_of_interest = ['VOIP']
    X_train_split, Y_train, X_val_split, Y_val, X_test_split, Y_test = read_csv(file_path, labels_of_interest=class_of_interest)
    X_train_split, Y_train = shuffle(X_train_split, Y_train, random_state=42)

    # Apply SMOTE
    if args.smote is not None:
      smote = SMOTEENN(sampling_strategy='auto', random_state=42)
      X_train_split, Y_train = smote.fit_resample(X_train_split, Y_train)
      print(f'Positives after SMOTE: {np.sum(Y_train)}')

    #
    weights = compute_sample_weight(class_weight='balanced', y=Y_train)
    print(f"Weights: {np.unique(weights)}")
    #
    METRIX = []
    for es in estimators:
      for mss in min_samples_split:
        for md in max_depth:
          for msl in min_samples_leaf:
            for ms in max_samples:
              for mf in max_features:
                # Create the model
                MODEL = RF(n_estimators=es, min_samples_split=mss, max_depth=md,
                    min_samples_leaf=msl, max_samples=ms, max_features=mf)
                MODEL.fit(X_train_split, Y_train, sample_weight=weights)
                #
                OUT_train = MODEL.predict(X_train_split)
                OUT_val   = MODEL.predict(X_val_split)
                OUT_test  = MODEL.predict(X_test_split)
                
                # Metrics
                acc_train, f1_train, prec_train, rec_train = compute_metrics(Y_train, OUT_train, split='Train')
                acc_val, f1_val, prec_val, rec_val = compute_metrics(Y_val, OUT_val, split='Val')
                acc_test, f1_test, prec_test, rec_test = compute_metrics(Y_test, OUT_test, split='Test')
                
                # CM
                cm_train = confusion_matrix(Y_train, OUT_train)
                cm_val   = confusion_matrix(Y_val, OUT_val)
                cm_test  = confusion_matrix(Y_test, OUT_test)
                #
                # print("Confusion matrix (train):\n", cm_train)
                # print("Confusion matrix (val):\n", cm_val)
                # print("Confusion matrix (test):\n", cm_test)
                #
                METRIX.append([acc_train, f1_train, prec_train, rec_train,
                               acc_val, f1_val, prec_val, rec_val,
                               acc_test, f1_test, prec_test, rec_test])
                #
                # Update best model if current is better
                if f1_val > best_val_score:
                    best_val_score = f1_val
                    best_idx = idx_sim
                    best_model = MODEL
                    # print(f"New best model found with: {es} estimators, {mss} minimum sample split",
                    #   f"{md} max depth, {msl} minimum sample per leaf, {ms} max sample count/percentage, "
                    #   f"{mf} max features. Val Mean UA: {best_val_score:.2f}")

                idx_sim += 1
    
    # Best scores:
    print("Best scores: ")
    print(METRIX[best_idx])
    METRIX = np.array(METRIX)

    #
    sim_list_idx = range(0, Nsim)
    sim_list_estimators = []
    sim_list_min_samples_split = []
    sim_list_max_depth = []
    sim_list_min_samples_leaf = []
    sim_list_max_samples = []
    sim_list_max_features = []
    for es in estimators:
      for mss in min_samples_split:
        for md in max_depth:
          for msl in min_samples_leaf:
            for ms in max_samples:
              for mf in max_features:
                sim_list_estimators.append(es)
                sim_list_min_samples_split.append(mss)
                sim_list_max_depth.append(md)
                sim_list_min_samples_leaf.append(msl)
                sim_list_max_samples.append(ms)
                sim_list_max_features.append(mf)

    df = pd.DataFrame({
        'SIM': sim_list_idx,
        'n_estimators': sim_list_estimators,
        'min_samples_split': sim_list_min_samples_split,
        'max_depth': sim_list_max_depth,
        'min_samples_leaf': sim_list_min_samples_leaf,
        'max_samples': sim_list_max_samples,
        'max_features': sim_list_max_features,

        'acc_train': METRIX[:, 0],
        'f1_train':  METRIX[:, 1],
        'prec_train': METRIX[:, 2],
        'rec_train':  METRIX[:, 3],

        'acc_val': METRIX[:, 4],
        'f1_val':  METRIX[:, 5],
        'prec_val': METRIX[:, 6],
        'rec_val':  METRIX[:, 7],

        'acc_test': METRIX[:, 8],
        'f1_test':  METRIX[:, 9],
        'prec_test': METRIX[:,10],
        'rec_test':  METRIX[:,11],
    })
    df = pd.DataFrame(df)
    csv_path = 'results'
    os.makedirs(csv_path, exist_ok=True)
    results_path = os.path.join(csv_path, 'RF_configs.csv')

    if os.path.exists(results_path):
        df.to_csv(results_path, mode='a', header=False, index=False)
    else:
        df.to_csv(results_path, index=False)