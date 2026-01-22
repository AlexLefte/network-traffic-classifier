import argparse
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.decomposition      import PCA
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.utils import shuffle
from imblearn.combine import SMOTEENN
from utils import read_csv


def compute_metrics(y_true, y_pred, split='Train'):
    """Compute classification metrics"""
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
    
    return accuracy, f1, precision, recall


# Define MLP Model
class MLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, dropout_rate):
        super(MLP, self).__init__()
        layers = []
        prev_size = input_size
        for size in hidden_sizes:
            layers.append(nn.Linear(prev_size, size))
            layers.append(nn.BatchNorm1d(size))
            layers.append(nn.LeakyReLU(negative_slope=0.1))
            layers.append(nn.Dropout(dropout_rate))
            prev_size = size
        layers.append(nn.Linear(prev_size, output_size))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


def run_cv_config(
    df,
    flow_labels,
    pca,
    hidden_sizes,
    dropout,
    labels_of_interest,
    device,
    n_folds=4,
    batch_size=512,
    num_epochs=200,
    early_stop_patience=20,
    use_smote=False,
    random_state=42
):
    """Run cross-validation for a specific MLP configuration"""
    sgkf = StratifiedGroupKFold(
        n_splits=n_folds,
        shuffle=True,
        random_state=random_state
    )

    X_idx = flow_labels.index.values
    y = flow_labels['Label'].values
    groups = flow_labels['flow_id'].values

    fold_metrics = []

    for fold_id, (tr, va) in tqdm(enumerate(sgkf.split(X_idx, y, groups)), desc="Running folds...", total=n_folds):
        print(f"\n--- Fold {fold_id + 1}/{n_folds} ---")
        
        train_flows = set(flow_labels.iloc[tr]['flow_id'])
        val_flows = set(flow_labels.iloc[va]['flow_id'])

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
        
        X_tr = df_tr.drop(columns=drop_cols).values
        y_tr = df_tr['binary_label'].values
        X_val = df_va.drop(columns=drop_cols).values
        y_val = df_va['binary_label'].values

        # Scale features
        # scaler = MinMaxScaler()
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr)
        X_val = scaler.transform(X_val)

        X_tr = np.nan_to_num(X_tr, nan=0)
        X_val = np.nan_to_num(X_val, nan=0)

        # Shuffle training data
        X_tr, y_tr = shuffle(X_tr, y_tr, random_state=random_state)

        if pca_comp != 'no_pca':
            pca = PCA(n_components=pca_comp)
            X_tr = pca.fit_transform(X_tr)
            X_val = pca.transform(X_val)

        # Apply SMOTE if needed
        if use_smote:
            smote = SMOTEENN(random_state=42)
            X_tr, y_tr = smote.fit_resample(X_tr, y_tr)

        # Compute class weights
        class_counts = np.bincount(y_tr)
        total_samples = len(y_tr)
        class_weights = torch.FloatTensor([
            total_samples / (len(class_counts) * class_counts[0]),
            total_samples / (len(class_counts) * class_counts[1])
        ]).to(device)

        # Create weighted sampler for balanced batches
        sample_weights = np.array([1.0/class_counts[y] for y in y_tr])
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )

        # Convert to tensors
        X_train_tensor = torch.tensor(X_tr, dtype=torch.float32).to(device)
        Y_train_tensor = torch.tensor(y_tr, dtype=torch.long).to(device)
        X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)
        Y_val_tensor = torch.tensor(y_val, dtype=torch.long).to(device)

        # Create DataLoaders
        train_dataset = TensorDataset(X_train_tensor, Y_train_tensor)
        val_dataset = TensorDataset(X_val_tensor, Y_val_tensor)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        # Initialize model
        input_size = X_tr.shape[1]
        output_size = 2
        model = MLP(input_size, hidden_sizes, output_size, dropout).to(device)
        
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

        # Training loop with early stopping
        best_val_f1 = -float('inf')
        epochs_no_improve = 0
        best_model_state = None

        for epoch in tqdm(range(num_epochs), desc=f"Running fold: {fold_id + 1}."):
            model.train()
            epoch_loss = 0.0
            for inputs, targets in train_loader:
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            # Validate every 5 epochs
            if epoch % 5 == 0:
                model.eval()
                val_preds = []
                val_true = []
                
                with torch.inference_mode():
                    for inputs, targets in val_loader:
                        outputs = model(inputs)
                        preds = torch.argmax(outputs, dim=1).cpu().numpy()
                        val_preds.extend(preds)
                        val_true.extend(targets.cpu().numpy())
                
                val_f1 = f1_score(val_true, val_preds, average='macro', zero_division=0)
                
                if val_f1 > best_val_f1:
                    best_val_f1 = val_f1
                    epochs_no_improve = 0
                    best_model_state = model.state_dict()
                else:
                    epochs_no_improve += 1

                if epochs_no_improve >= early_stop_patience:
                    break

        # Load best model
        if best_model_state is not None:
            model.load_state_dict(best_model_state)

        # Final evaluation
        model.eval()
        with torch.inference_mode():
            # Train predictions
            train_outputs = []
            train_targets = []
            for inputs, targets in train_loader:
                outputs = model(inputs).cpu().numpy()
                train_outputs.append(outputs)
                train_targets.append(targets.cpu().numpy())
            train_outputs = np.concatenate(train_outputs)
            train_targets = np.concatenate(train_targets)
            train_predictions = np.argmax(train_outputs, axis=1)

            # Validation predictions
            val_outputs = []
            val_targets = []
            for inputs, targets in val_loader:
                outputs = model(inputs).cpu().numpy()
                val_outputs.append(outputs)
                val_targets.append(targets.cpu().numpy())
            val_outputs = np.concatenate(val_outputs)
            val_targets = np.concatenate(val_targets)
            val_predictions = np.argmax(val_outputs, axis=1)

        # Compute metrics
        acc_train, f1_train, prec_train, rec_train = compute_metrics(
            train_targets, train_predictions, split=f"Fold {fold_id} Train"
        )
        acc_val, f1_val, prec_val, rec_val = compute_metrics(
            val_targets, val_predictions, split=f"Fold {fold_id} Val"
        )

        fold_metrics.append([
            acc_train, f1_train, prec_train, rec_train,
            acc_val, f1_val, prec_val, rec_val
        ])

    fold_metrics = np.array(fold_metrics)

    return (
        # Train
        fold_metrics[:, 0].mean(),
        fold_metrics[:, 1].mean(),
        fold_metrics[:, 2].mean(),
        fold_metrics[:, 3].mean(),
        # Validation
        fold_metrics[:, 4].mean(),
        fold_metrics[:, 5].mean(),
        fold_metrics[:, 6].mean(),
        fold_metrics[:, 7].mean()
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_file", type=str, required=True)
    parser.add_argument("--smote", action='store_true')
    args = parser.parse_args()

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Running on: {device}')

    # Hyperparameters
    pca = ['no_pca', 0.95]
    hidden_layers = [
        [128, 128, 128, 64, 32, 16], 
        [32, 32, 16], 
        [32, 16], 
        [128, 64], 
        [64, 32, 32],
        [128, 64, 32, 16],
        [64, 32],
        [256, 128, 64, 32, 16],
        [256, 128, 64],
        [128, 128, 64, 32],
        [64, 64, 32],
        # [512, 256, 128, 64, 32, 16],
        [512, 256, 128],
        [256, 256, 128, 64],
        [128, 64, 32],
        # [512, 256, 128, 64],
        [256, 128, 64, 32],
        [64, 32, 16],
        [128, 64, 32, 16, 8],
        # [256, 128, 64, 32, 16, 8]
    ]
    dropouts = [0.3, 0.5]

    # class_of_interest = [14, 15, 16, 17]
    class_of_interest = ['VOIP']

    # Read data
    df, flow_labels = read_csv(
        args.csv_file
    )

    idx_sim = 0
    best_f1 = -np.inf
    best_idx = -1

    # Grid search
    for smote in [True]:
        for pca_comp in pca:
            for hidden_sizes in hidden_layers:
                    for dropout in dropouts:
                        print(f"\n{'='*60}")
                        print(f"Config {idx_sim}: hidden={hidden_sizes}, dropout={dropout}")
                        print(f"{'='*60}")
                    train_acc, train_f1, train_prec, train_rec, \
                        val_acc, val_f1, val_prec, val_rec = run_cv_config(
                        df,
                        flow_labels,
                        pca = pca_comp,
                        hidden_sizes=hidden_sizes,
                        dropout=dropout,
                        labels_of_interest=class_of_interest,
                        device=device,
                        n_folds=4,
                        batch_size=256,
                        num_epochs=200,
                        early_stop_patience=20,
                        use_smote=smote
                    )

                    # Creează DataFrame temporar pentru experimentul curent
                    df_exp = pd.DataFrame([{
                        'SIM': idx_sim,
                        'hidden_layers': '_'.join(map(str, hidden_sizes)),
                        'dropout': dropout,
                        'smote': smote,
                        'train_acc': train_acc,
                        'train_f1': train_f1,
                        'train_prec': train_prec,
                        'train_rec': train_rec,
                        'val_acc': val_acc,
                        'val_f1': val_f1,
                        'val_prec': val_prec,
                        'val_rec': val_rec,
                        'pca': pca_comp
                    }])

                    # Salvează append în CSV după fiecare experiment
                    os.makedirs("results", exist_ok=True)
                    out_path = f"results/MLP_CV_results_standard_scaler_{args.csv_file.split('/')[-1].replace('.csv','')}.csv"
                    if os.path.exists(out_path):
                        df_exp.to_csv(out_path, mode='a', header=False, index=False)
                    else:
                        df_exp.to_csv(out_path, index=False)

                    # Check best model
                    if val_f1 > best_f1:
                        best_f1 = val_f1
                        best_idx = idx_sim
                        print(f"\nNEW BEST: SIM={idx_sim} | F1={val_f1:.3f}")

                    idx_sim += 1    

    print("\nDONE GRID SEARCH")
    print(f"{'='*60}")

    # Cea mai bună configurație
    if os.path.exists(out_path):
        df_all = pd.read_csv(out_path)
        # Găsește rândul cu cel mai bun val_f1
        best_row = df_all.loc[df_all['val_f1'].idxmax()]
        print("Best config overall:")
        print(best_row)
        print(f"{'='*60}")
    else:
        print("CSV file not found, cannot show best config")