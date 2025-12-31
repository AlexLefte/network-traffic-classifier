import os
import pickle
import argparse
import numpy  as np
import pandas as pd
#
from sklearn.svm                import SVC
from sklearn.decomposition      import PCA
from sklearn.utils              import shuffle
from sklearn.metrics            import accuracy_score, f1_score, confusion_matrix, precision_recall_fscore_support
from utils                      import read_csv
from sklearn.utils.class_weight import compute_sample_weight
#
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_file", type=str, required=True)
    args = parser.parse_args()
    #
    # Datasets
    root_path = ""
    file_path = args.csv_file
    #
    # Hyperparameters
    PCA_components = ['no_pca', 5, 10, 15, 20, 25]
    SVM_kernels = ['rbf']
    Cs = [20, 10, 1, 5e-1, 1e-1]
    gammas = ['scale']
    #
    # Number of simulations and metrix array definition
    Nsim = len(PCA_components) * len(SVM_kernels) * len(Cs) * len(gammas)
    #
    idx_sim = 0
    METRIX_ = np.zeros((Nsim, 4))
    #
    # Save best model for the current split
    best_model = None
    best_val_score = -float('inf')
    #
    # Retrieve the csv paths
    file_path = os.path.join(root_path, f"flows.csv")
    #
    # Read and shffle the data
    X_train_split, Y_train, X_val_split, Y_val, X_test_split, Y_test = read_csv(file_path)
    X_train_split, Y_train = shuffle(X_train_split, Y_train, random_state=42)
    #
    weights = compute_sample_weight(class_weight='balanced', y=Y_train)
    print(f"Weights: {np.unique(weights)}")
    #
    for pca_comp in PCA_components:
        for SVM_kernel in SVM_kernels:
            for C in Cs:
                for gamma in gammas:
                    METRIX = []
                    #
                    # Perform PCA for feature reduction
                    if pca_comp != 'no_pca':
                        pca = PCA(n_components=pca_comp)
                        X_train_split = pca.fit_transform(X_train_split)
                        X_val_split = pca.transform(X_val_split)
                        X_test_split = pca.transform(X_test_split)
                    #
                    MODEL = SVC(C=C, kernel=SVM_kernel, tol=1.0)
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
                    if f1_val > best_val_score:
                        best_val_score = f1_val
                        best_model = MODEL
                        print(f"New best model found with PCA: {pca_comp}, Kernel: {SVM_kernel}, C: {C}, gamma {gamma}, Val Mean UA: {best_val_score:.2f}")
                    #
                    idx_sim += 1
    #
    # Save best model
    exp_name = 'svc'
    model_path = os.path.join(root_path, f"models/SVC/{exp_name}")
    os.makedirs(model_path, exist_ok=True)
    model_path = os.path.join(model_path, f"best_svc_configs.pkl")
    #
    # Read the data
    X_train_split, Y_train, X_val_split, Y_val, X_test_split, Y_test = read_csv(file_path)
    X_train_split = np.concatenate((X_train_split, X_val_split, X_test_split), axis=0)
    Y_train = np.concatenate((Y_train, Y_val, Y_test), axis=0)
    #
    # Shuffle data
    X_train_split, Y_train = shuffle(X_train_split, Y_train, random_state=42)
    #
    # Train the Model on the entire dataset and save
    best_model.fit(X_train_split, Y_train)
    with open(model_path, "wb") as f:
        pickle.dump(best_model, f)
    #
    sim_list_idx = range(0, Nsim)
    sim_list_pca = []
    sim_list_SVM_kernels = []
    sim_list_Cs = []
    sim_list_gammas = []
    for pca_comp in PCA_components:
        for SVM_kernel in SVM_kernels:
            for C in Cs:
                for gamma in gammas:
                    sim_list_pca.append(pca_comp)
                    sim_list_SVM_kernels.append(SVM_kernel)
                    sim_list_Cs.append(C)
                    sim_list_gammas.append(gamma)
    #
    df_dict = { k:v for (k, v) in zip(['SIM', 'PCA_comp', 'Kernel', 'C',
                                        'Acc_train [%]', 'F1_train [%]',
                                        'Acc_val [%]', 'F1_val [%]', 'g'],
                                        [sim_list_idx, sim_list_pca, sim_list_SVM_kernels,
                                        sim_list_Cs,
                                        METRIX_[:,0], METRIX_[:,1],
                                        METRIX_[:,2], METRIX_[:,3], sim_list_gammas])}
    #
    df = pd.DataFrame(df_dict)
    csv_path = os.path.join(root_path, 'results')
    os.makedirs(csv_path, exist_ok=True)
    results_path = os.path.join(csv_path, f'SVC_configs.csv')
    #
    if os.path.exists(results_path):
        df.to_csv(results_path, mode='a', header=False, index=False)
    else:
        df.to_csv(results_path, index=False)