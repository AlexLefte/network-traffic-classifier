import os
import pickle
import argparse
import numpy  as np
import pandas as pd
#
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm                import SVC
from sklearn.decomposition      import PCA
from sklearn.utils              import shuffle
from sklearn.metrics            import confusion_matrix
from utils                      import read_csv_and_split, compute_metrics, read_csv_multiclass, read_arff_multiclass, read_csv
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.model_selection import StratifiedGroupKFold
from imblearn.combine import SMOTEENN


def run_cv_config(
    df,
    flow_labels,
    pca_comp,
    kernel,
    C,
    gamma,
    labels_of_interest, 
    n_folds=4,
    use_smote=False,
    random_state=42
):
    sgkf = StratifiedGroupKFold(
        n_splits=n_folds,
        shuffle=True,
        random_state=random_state
    )

    X_idx = flow_labels.index.values
    y = flow_labels['Label'].values
    groups = flow_labels['flow_id'].values

    fold_metrics = []

    for fold_id, (tr, va) in enumerate(sgkf.split(X_idx, y, groups)):
        train_flows = set(flow_labels.iloc[tr]['flow_id'])
        val_flows   = set(flow_labels.iloc[va]['flow_id'])

        df_tr = df[df['flow_id'].isin(train_flows)].copy()
        df_va = df[df['flow_id'].isin(val_flows)].copy()

        # Binarize after split
        df_tr['binary_label'] = df_tr['Label'].apply(
            lambda x: 1 if x in labels_of_interest else 0
        )
        df_va['binary_label'] = df_va['Label'].apply(
            lambda x: 1 if x in labels_of_interest else 0
        )

        # Drop unnecessary columns
        drop_cols = ['Label', 'binary_label', 'flow_id', 'flow_id_init', 'file']
        drop_cols = [c for c in drop_cols if c in df_tr.columns]
        X_tr = df_tr.drop(columns=drop_cols)
        y_tr = df_tr['binary_label']
        X_val = df_va.drop(columns=drop_cols)
        y_val = df_va['binary_label']

        scaler = MinMaxScaler()
        X_tr = scaler.fit_transform(X_tr)
        X_val = scaler.transform(X_val)

        X_tr = np.nan_to_num(X_tr, nan=-1)
        X_val = np.nan_to_num(X_val, nan=-1)

        if use_smote:
            smote = SMOTEENN(random_state=42)
            X_tr, y_tr = smote.fit_resample(X_tr, y_tr)

        if pca_comp != 'no_pca':
            pca = PCA(n_components=pca_comp)
            X_tr = pca.fit_transform(X_tr)
            X_val = pca.transform(X_val)

        weights = compute_sample_weight('balanced', y_tr)

        model = SVC(C=C, kernel=kernel, gamma=gamma, tol=1.0)
        model.fit(X_tr, y_tr, sample_weight=weights)

        # Train results
        y_pred_train = model.predict(X_tr)
        acc_train, f1_train, prec_train, rec_train = compute_metrics(
            y_tr, y_pred_train, split=f"Fold {fold_id} Train"
        )
 
        # Val results
        y_pred_val = model.predict(X_val)
        acc_val, f1_val, prec_val, rec_val = compute_metrics(
            y_val, y_pred_val, split=f"Fold {fold_id}"
        )

        fold_metrics.append([
            acc_train, f1_train, prec_train, rec_train,
            acc_val, f1_val, prec_val, rec_val])

    fold_metrics = np.array(fold_metrics)

    return (
        # Train
        fold_metrics[:,0].mean(),
        fold_metrics[:,1].mean(),
        fold_metrics[:,2].mean(),
        fold_metrics[:,3].mean(),

        # Validation
        fold_metrics[:,4].mean(),
        fold_metrics[:,5].mean(),
        fold_metrics[:,6].mean(),
        fold_metrics[:,7].mean()
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_file", type=str, required=True)
    parser.add_argument("--smote", action='store_true')
    args = parser.parse_args()

    PCA_components = ['no_pca']
    SVM_kernels = ['rbf']
    Cs = [100, 50, 20, 10, 1, 0.1, 0.01, 0.001]
    gammas = ['scale']

    class_of_interest = [14, 15, 16, 17]

    df, flow_labels = read_csv(
        args.csv_file,
        labels_of_interest=class_of_interest
    )

    METRIX = []
    sim_list = []

    idx_sim = 0
    best_f1 = -np.inf
    best_idx = -1

    for pca_comp in PCA_components:
        for kernel in SVM_kernels:
            for C in Cs:
                for gamma in gammas:

                    train_acc, train_f1, train_prec, train_rec, \
                        val_acc, val_f1, val_prec, val_rec = run_cv_config(
                        df,
                        flow_labels,
                        pca_comp,
                        kernel,
                        C,
                        gamma,
                        labels_of_interest=class_of_interest, 
                        n_folds=4,
                        use_smote=args.smote
                    )

                    METRIX.append([
                        train_acc, train_f1, train_prec, train_rec,
                        val_acc, val_f1, val_prec, val_rec
                    ])

                    sim_list.append([
                        idx_sim, pca_comp, kernel, C, gamma
                    ])

                    if val_f1 > best_f1:
                        best_f1 = val_f1
                        best_idx = idx_sim
                        print(f"\nNEW BEST: SIM={idx_sim} | F1={val_f1:.3f}")

                    idx_sim += 1

    METRIX = np.array(METRIX)
    sim_list = np.array(sim_list, dtype=object)

    df_results = pd.DataFrame({
        'SIM': sim_list[:,0],
        'PCA_components': sim_list[:,1],
        'kernel': sim_list[:,2],
        'C': sim_list[:,3],
        'gamma': sim_list[:,4],
        'smote': args.smote,

        # Train stats
        'train_acc': METRIX[:,0],
        'train_f1':  METRIX[:,1],
        'train_prec': METRIX[:,2],
        'train_rec':  METRIX[:,3],

        # Val stats
        'val_acc': METRIX[:,4],
        'val_f1':  METRIX[:,5], 
        'val_prec': METRIX[:,6],
        'val_rec':  METRIX[:,7],
    })

    os.makedirs("results", exist_ok=True)
    out_path = "results/SVC_CV_results.csv"
    if os.path.exists(out_path):
        df_results.to_csv(out_path, mode='a', header=False, index=False)
    else:
        df_results.to_csv(out_path, index=False)

    print("\nDONE")
    print("Best config:")
    print(df_results.iloc[best_idx])