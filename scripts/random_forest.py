import numpy as np
from sklearn import datasets as SKdata
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.utils import shuffle
import pandas as pd
from time import time
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
import pickle
import seaborn as sns
import os
from utils import read_csv


if __name__ == '__main__':
  # Datasets
  root_path = ""
  splits = [0, 1, 2, 3, 4]  # From 0 to 4 for each split
  windows = [300] # Windows
  Kclass = 3

  # Hyperparameters
  estimators = list(range(5, 20, 5))
  min_samples_split = [0.05, 0.1, 0.15, 0.2]
  max_depth = list(range(3, 10))
  min_samples_leaf = [0.05, 0.1]
  max_samples = [0.3, 0.4, 0.5, 0.6, 0.7]
  max_features = ['sqrt', 'log2']

  # Compute no simulations to perform
  Nsim = len(estimators)*len(min_samples_split)*len(max_depth)*len(min_samples_leaf)*len(max_samples)*len(max_features)

  for window in windows:
    idx_sim = 0
    METRIX_ = np.zeros((Nsim, 4))

    # Retrieve the csv paths
    file_paths = [f"" for i in splits]

    # Save best model for the current split
    best_model = None
    best_val_score = -float('inf')

    for es in estimators:
      for mss in min_samples_split:
        for md in max_depth:
          for msl in min_samples_leaf:
            for ms in max_samples:
              for mf in max_features:
                METRIX = []
                for split_index, file_path in enumerate(file_paths):
                  # Read the data
                  X_train_split, Y_train, X_val_split, Y_val = read_csv(file_path)

                  # Shuffle data
                  X_train_split, Y_train = shuffle(X_train_split, Y_train, random_state=42)

                  # Create the model
                  MODEL = RF(n_estimators=es, min_samples_split=mss, max_depth=md,
                      min_samples_leaf=msl, max_samples=ms, max_features=mf)
                  start = time()
                  MODEL.fit(X_train_split, Y_train)
                  end = time()

                  print('Training time: %.2f sec' % (end-start))
                  OUT_train = MODEL.predict(X_train_split)
                  OUT_val = MODEL.predict(X_val_split)

                  # Train metrics
                  acc_train = accuracy_score(Y_train, OUT_train)
                  f1_train = f1_score(Y_train, OUT_train, average='weighted')
                  print(f'acc (train) = {acc_train}. f1 (train) = {f1_train}')

                  acc_val = accuracy_score(Y_val, OUT_val)
                  f1_val = f1_score(Y_val, OUT_val, average='weighted')
                  print(f'acc (val) = {acc_val}. f1 (val) = {f1_val}')
                  METRIX += [acc_train, f1_train, acc_val, f1_val]

                # -> Cross-validation results:
                acc_train_avg = f1_train_avg = acc_val_avg = f1_val_avg = 0
                L = len(METRIX)
                for i in range(0, L, 4):
                    acc_train_avg += METRIX[i]
                acc_train_avg = np.round(acc_train_avg/5, decimals=2)
                for i in range(1, L, 4):
                    f1_train_avg += METRIX[i]
                f1_train_avg = np.round(f1_train_avg/5, decimals=2)
                for i in range(2, L, 4):
                    acc_val_avg += METRIX[i]
                acc_val_avg = np.round(acc_val_avg/5, decimals=2)
                for i in range(3, L, 4):
                    f1_val_avg += METRIX[i]
                f1_val_avg = np.round(f1_val_avg/5, decimals=2)
                print(f'Acc avg (train) = {acc_train_avg}. F1 avg (train) = {f1_train_avg}')
                print(f'Acc avg (val) = {acc_val_avg}. F1 avg (val) = {f1_val_avg}\n')
                METRIX_[idx_sim,:] = [acc_train_avg, f1_train_avg,
                                    acc_val_avg, f1_val_avg]

                # Update best model if current is better
                if f1_val_avg > best_val_score:
                    best_val_score = f1_val_avg
                    best_model = MODEL
                    print(f"New best model found with: {es} estimators, {mss} minimum sample split",
                      f"{md} max depth, {msl} minimum sample per leaf, {ms} max sample count/percentage, "
                      f"{mf} max features. Val Mean UA: {best_val_score:.2f}")

                idx_sim += 1

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

      # Save best model
      rf_path = 'models/RF/'
      os.makedirs(rf_path, exist_ok=True)
      model_path = os.path.join(rf_path, f"best_rf_{window}_frequency_included.pkl")
      os.makedirs(os.path.dirname(file_path), exist_ok=True)
      # Read the data
      X_train_split, Y_train, X_val_split, Y_val = read_csv(file_path)
      X_train_split = np.concatenate((X_train_split, X_val_split), axis=0)
      Y_train = np.concatenate((Y_train, Y_val), axis=0)
      # Shuffle data
      X_train_split, Y_train = shuffle(X_train_split, Y_train, random_state=42)
      # Train the Model on the entire dataset and save
      best_model.fit(X_train_split, Y_train)
      with open(model_path, "wb") as f:
          pickle.dump(best_model, f)

      df_dict = { k:v for (k, v) in zip(['SIM', 'Es', 'Mss', 'Md', 'Msl', 'Ms', 'Mf',
                                        'Acc_train [%]', 'F1_train [%]',
                                        'Acc_val [%]', 'F1_val [%]'],
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
      results_path = os.path.join(csv_path, f'RF_{window}.csv')
      # Verifică dacă fișierul există
      if os.path.exists(results_path):
          # Scrie în fișier folosind append (fără header)
          df.to_csv(results_path, mode='a', header=False, index=False)
      else:
          # Scrie în fișier cu header (pentru prima scriere)
          df.to_csv(results_path, index=False)