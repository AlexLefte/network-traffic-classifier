import os
import pickle
import argparse
import numpy  as np
import pandas as pd
#
from sklearn.svm                import SVC
from sklearn.decomposition      import PCA
from sklearn.utils              import shuffle
from sklearn.metrics            import confusion_matrix
from utils                      import read_csv_and_split, compute_metrics, read_csv_multiclass, read_arff_multiclass
from sklearn.utils.class_weight import compute_sample_weight
from imblearn.combine import SMOTEENN
#
            
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_file", type=str, required=True)
    parser.add_argument("--smote", action='store_true')
    args = parser.parse_args()
    
    # Datasets
    root_path = ""
    file_path = args.csv_file
    
    # Hyperparameters
    PCA_components = ['no_pca', 0.95]
    SVM_kernels = ['rbf']
    Cs = [100, 50, 20, 10, 1, 0.1, 0.01, 0.001]
    gammas = ['scale']
    
    # Number of simulations and metrix array definition
    Nsim = len(PCA_components) * len(SVM_kernels) * len(Cs) * len(gammas)
    idx_sim = 0
    
    # Save best model for the current split
    best_model = None
    best_val_score = -float('inf')
    
    # Retrieve the csv paths
    file_path = os.path.join(root_path, file_path)
    
    # Read and shffle the data
    class_of_interest = [14, 15, 16, 17]
    
    # class_of_interest = ['STREAMING']
    X_train_split, Y_train, X_val_split, Y_val, X_test_split, Y_test = read_csv_and_split(file_path, labels_of_interest=class_of_interest)
    X_train_split, Y_train = shuffle(X_train_split, Y_train, random_state=42)
    print(f'Positives before SMOTE: {np.sum(Y_train)}')

    # Apply SMOTEENN
    if args.smote is not None:
      smote = SMOTEENN(sampling_strategy='auto', random_state=42)
      X_train_split, Y_train = smote.fit_resample(X_train_split, Y_train)
      print(f'Positives after SMOTE: {np.sum(Y_train)}')

    #
    weights = compute_sample_weight(class_weight='balanced', y=Y_train)
    print(f"Weights: {np.unique(weights)}")
    #
    METRIX = []
    for pca_comp in PCA_components:
        for SVM_kernel in SVM_kernels:
            for C in Cs:
                for gamma in gammas:
                    
                    #
                    # Perform PCA for feature reduction
                    if pca_comp != 'no_pca':
                        pca = PCA(n_components=pca_comp)
                        X_train_split = pca.fit_transform(X_train_split)
                        X_val_split = pca.transform(X_val_split)
                        X_test_split = pca.transform(X_test_split)
                        print(X_train_split.shape)
                    #
                    MODEL = SVC(C=C, kernel=SVM_kernel, tol=1.0)
                    MODEL.fit(X_train_split, Y_train, sample_weight=weights)
                    #
                    OUT_train = MODEL.predict(X_train_split)
                    OUT_val   = MODEL.predict(X_val_split)
                    OUT_test  = MODEL.predict(X_test_split)
                    #
                    # Metrics
                    acc_train, f1_train, prec_train, rec_train = compute_metrics(Y_train, OUT_train, split='Train')
                    acc_val, f1_val, prec_val, rec_val = compute_metrics(Y_val, OUT_val, split='Val')
                    acc_test, f1_test, prec_test, rec_test = compute_metrics(Y_test, OUT_test, split='Test')

                    cm_train = confusion_matrix(Y_train, OUT_train)
                    cm_val   = confusion_matrix(Y_val, OUT_val)
                    cm_test  = confusion_matrix(Y_test, OUT_test)
                    #
                    print("Confusion matrix (train):\n", cm_train)
                    print("Confusion matrix (val):\n", cm_val)
                    print("Confusion matrix (test):\n", cm_test)
                    #
                    METRIX.append([acc_train, f1_train, prec_train, rec_train,
                               acc_val, f1_val, prec_val, rec_val,
                               acc_test, f1_test, prec_test, rec_test])
                    #
                    if f1_val > best_val_score:
                        best_val_score = f1_val
                        best_model = MODEL
                        best_idx = idx_sim
                        print(f"New best model found with PCA: {pca_comp}, Kernel: {SVM_kernel}, C: {C}, gamma {gamma}, Val Mean UA: {best_val_score:.2f}")
                    #
                    idx_sim += 1
    
    # Best scores:
    print("Best scores: ")
    print(METRIX[best_idx])
    METRIX = np.array(METRIX)

    # Create csv
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
    df_dict = pd.DataFrame({
        'SIM': sim_list_idx,
        'PCA_components': sim_list_pca,
        'kernel': sim_list_SVM_kernels,
        'C': sim_list_Cs,
        'gamma': sim_list_gammas,

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

    #
    df = pd.DataFrame(df_dict)
    csv_path = os.path.join(root_path, 'results')
    os.makedirs(csv_path, exist_ok=True)

    results_path = os.path.join(csv_path, 'SVC_configs.csv')

    if os.path.exists(results_path):
        df.to_csv(results_path, mode='a', header=False, index=False)
    else:
        df.to_csv(results_path, index=False)