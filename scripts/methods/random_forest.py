import numpy  as np
import pandas as pd
import os
import pickle
import matplotlib.pyplot as plt
#
from sklearn.ensemble           import RandomForestClassifier as RF
from sklearn.utils              import shuffle
from sklearn.metrics            import accuracy_score, f1_score, confusion_matrix, precision_recall_fscore_support
from utils                      import read_csv
from sklearn.utils.class_weight import compute_sample_weight
#
#
if __name__ == '__main__':
    #
    # Datasets
    root_path = ""
    file_path = "flows.csv"
    #
    # Hyperparameters
    estimators = list(range(5, 20, 5))
    min_samples_split = [0.05, 0.1, 0.15, 0.2]
    max_depth = list(range(3, 10))
    min_samples_leaf = [0.05, 0.1]
    max_samples = [0.3, 0.4, 0.5, 0.6, 0.7]
    max_features = ['sqrt', 'log2']
    #
    # Compute no simulations to perform
    Nsim = len(estimators)*len(min_samples_split)*len(max_depth)*len(min_samples_leaf)*len(max_samples)*len(max_features)
    #
    idx_sim = 0
    METRIX_ = np.zeros((Nsim, 4))
    #
    # Save best model for the current split
    best_model = None
    best_val_score = -float('inf')
    #
    # Read and shuffle the data
    X_train_split, Y_train, X_val_split, Y_val, X_test_split, Y_test = read_csv(file_path)
    X_train_split, Y_train = shuffle(X_train_split, Y_train, random_state=42)
    #
    weights = compute_sample_weight(class_weight='balanced', y=Y_train)
    # print(f"Weights: {np.unique(weights)}")
    #
    for es in estimators:
      for mss in min_samples_split:
        for md in max_depth:
          for msl in min_samples_leaf:
            for ms in max_samples:
              for mf in max_features:
                METRIX = []
                #
                # Create the model
                MODEL = RF(n_estimators=es, min_samples_split=mss, max_depth=md,
                    min_samples_leaf=msl, max_samples=ms, max_features=mf)
                MODEL.fit(X_train_split, Y_train, sample_weight=weights)
                #
                OUT_train = MODEL.predict(X_train_split)
                OUT_val   = MODEL.predict(X_val_split)
                OUT_test  = MODEL.predict(X_test_split)
                #
                # Train metrics
                acc_train = accuracy_score(Y_train, OUT_train)
                # f1_train = f1_score(Y_train, OUT_train, average='weighted')
                precision_train, recall_train, f1_train, _ = precision_recall_fscore_support(y_true=Y_train, y_pred=OUT_train, average='weighted')
                print(f'acc (train) = {acc_train}. f1 (train) = {f1_train}. precision (train) = {precision_train}. recall (train) = {recall_train}')
                #
                acc_val = accuracy_score(Y_val, OUT_val)
                # f1_val = f1_score(Y_val, OUT_val, average='weighted')
                precision_val, recall_val, f1_val, _ = precision_recall_fscore_support(y_true=Y_train, y_pred=OUT_train, average='weighted')
                print(f'acc (val) = {acc_val}. f1 (val) = {f1_val}. precision (val) = {precision_val}. recall (val) = {recall_val}')
                #
                acc_test = accuracy_score(Y_test, OUT_test)
                # f1_test = f1_score(Y_test, OUT_test, average='weighted')
                precision_test, recall_test, f1_test, _ = precision_recall_fscore_support(y_true=Y_train, y_pred=OUT_train, average='weighted')
                print(f'acc (test) = {acc_test}. f1 (test) = {f1_test}. precision (test) = {precision_test}. recall (test) = {recall_test}')
                #
                cm_train = confusion_matrix(Y_train, OUT_train)
                cm_val   = confusion_matrix(Y_val, OUT_val)
                cm_test  = confusion_matrix(Y_test, OUT_test)
                #
                print("Confusion matrix (train):\n", cm_train)
                print("Confusion matrix (val):\n", cm_val)
                print("Confusion matrix (test):\n", cm_test)
                #
                METRIX += [acc_train, f1_train, acc_val, f1_val, acc_test, f1_test]
                #
                # Update best model if current is better
                if f1_val > best_val_score:
                    best_val_score = f1_val
                    best_model = MODEL
                    print(f"New best model found with: {es} estimators, {mss} minimum sample split",
                      f"{md} max depth, {msl} minimum sample per leaf, {ms} max sample count/percentage, "
                      f"{mf} max features. Val Mean UA: {best_val_score:.2f}")

                idx_sim += 1
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
    #
    # Save best model
    rf_path = r"models\RF"
    os.makedirs(rf_path, exist_ok=True)

    df_dict = { k:v for (k, v) in zip(['SIM', 'Es', 'Mss', 'Md', 'Msl', 'Ms', 'Mf',
                                      'acc_train [%]', 'f1_train [%]',
                                      'acc_val [%]', 'f1_val [%]'],
                                      [sim_list_idx,
                                      sim_list_estimators,
                                      sim_list_min_samples_split,
                                      sim_list_max_depth,
                                      sim_list_min_samples_leaf,
                                      sim_list_max_samples,
                                      sim_list_max_features,
                                      METRIX_[:,0], METRIX_[:,1],
                                      METRIX_[:,2], METRIX_[:,3]]) }
    df = pd.DataFrame(df_dict)
    csv_path = os.path.join(root_path, 'results')
    os.makedirs(csv_path, exist_ok=True)
    results_path = os.path.join(csv_path, f'RF_configs.csv')
    #
    if os.path.exists(results_path):
        df.to_csv(results_path, mode='a', header=False, index=False)
    else:
        df.to_csv(results_path, index=False)