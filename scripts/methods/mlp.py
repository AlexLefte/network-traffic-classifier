import numpy as np
import pandas as pd
import argparse
import os
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from utils import read_csv_and_split


def compute_metrics(y_true, y_pred, split='Train'):
    """Compute classification metrics"""
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
    
    # Per-class metrics
    f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
    prec_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
    rec_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
    
    unique_classes = np.unique(y_true)
    print(f"\n{split} metrics per class:")
    for i, cls in enumerate(unique_classes):
        support = np.sum(y_true == cls)
        print(f"Class {cls}: precision={prec_per_class[i]:.3f}, recall={rec_per_class[i]:.3f}, "
              f"f1={f1_per_class[i]:.3f}, support={support}")
    
    print(f"Overall accuracy: {accuracy:.3f}")
    
    return accuracy, f1, precision, recall


# Define MLP Model - FIXED: Removed Softmax
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
        # NO SOFTMAX - CrossEntropyLoss expects raw logits
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


def plot_confusion_matrix(y_true, y_pred, title, filename):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues)
    plt.title(title)
    plt.savefig(filename)
    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_file", type=str, required=True)
    args = parser.parse_args()

    # Datasets
    root_path = ""
    file_path = args.csv_file

    # Choose device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Running on: ' + str(device))

    # Save best model
    best_model = None
    best_val_score = -float('inf')

    # SIMPLIFIED configurations for faster testing
    hidden_layers = [[128, 128, 128, 64, 32, 16], 
                     [32, 32, 16], 
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
                     [512, 256, 128, 64, 32, 16],
                     [512, 256, 128],
                     [256, 256, 128, 64],
                     [128, 64, 32],
                     [512, 256, 128, 64],
                     [256, 128, 64, 32],
                     [64, 32, 16],
                     [128, 64, 32, 16, 8],
                     [256, 128, 64, 32, 16, 8]]
    #
    dropouts = [0.3]
    output_size = 2
    batch_size = 256
    num_epochs = 200
    input_size = 29

    Nsim = len(hidden_layers) * len(dropouts)
    idx_sim = 0
    METRIX = np.zeros((Nsim, 12))

    # Read and shuffle data
    X_train_split, Y_train, X_val_split, Y_val, X_test_split, Y_test = read_csv_and_split(
        file_path, labels_of_interest=[14, 15, 16, 17]
    )
    X_train_split, Y_train = shuffle(X_train_split, Y_train, random_state=42)

    # FIX 1: Compute class weights for imbalanced data
    class_counts = np.bincount(Y_train)
    total_samples = len(Y_train)
    class_weights = torch.FloatTensor([
        total_samples / (len(class_counts) * class_counts[0]),
        total_samples / (len(class_counts) * class_counts[1])
    ]).to(device)
    
    print(f"\nClass distribution:")
    print(f"  Class 0: {class_counts[0]} ({100*class_counts[0]/total_samples:.1f}%)")
    print(f"  Class 1: {class_counts[1]} ({100*class_counts[1]/total_samples:.1f}%)")
    print(f"\nClass weights: {class_weights.cpu().numpy()}")

    # FIX 2: Create weighted sampler for balanced batches
    sample_weights = np.array([1.0/class_counts[y] for y in Y_train])
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )

    for hidden_sizes in hidden_layers:
        for dropout in dropouts:
            print(f"\n{'='*60}")
            print(f"Config: hidden={hidden_sizes}, dropout={dropout}")
            print(f"{'='*60}")

            # Define the tensors
            X_train_tensor = torch.tensor(np.array(X_train_split), dtype=torch.float32).to(device)
            Y_train_tensor = torch.tensor(np.array(Y_train), dtype=torch.long).to(device)
            X_val_tensor = torch.tensor(np.array(X_val_split), dtype=torch.float32).to(device)
            Y_val_tensor = torch.tensor(np.array(Y_val), dtype=torch.long).to(device)
            X_test_tensor = torch.tensor(np.array(X_test_split), dtype=torch.float32).to(device)
            Y_test_tensor = torch.tensor(np.array(Y_test), dtype=torch.long).to(device)

            # Create DataLoaders
            train_dataset = TensorDataset(X_train_tensor, Y_train_tensor)
            val_dataset = TensorDataset(X_val_tensor, Y_val_tensor)
            test_dataset = TensorDataset(X_test_tensor, Y_test_tensor)

            # FIX 3: Use weighted sampler instead of random shuffle
            train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

            # Model, Loss, Optimizer
            MODEL = MLP(input_size, hidden_sizes, output_size, dropout).to(device)
            
            # FIX 4: Use class weights in loss function
            criterion = nn.CrossEntropyLoss(weight=class_weights)
            
            # FIX 5: Lower weight decay (was too high at 1e-2)
            optimizer = torch.optim.Adam(MODEL.parameters(), lr=0.001, weight_decay=1e-4)

            # Training Loop
            train_losses = []
            val_losses = []
            val_f1_scores = []  # Track F1 for early stopping

            # Early stopping parameters
            early_stop_patience = 20
            best_val_f1 = -float('inf')
            epochs_no_improve = 0
            best_model_state = None

            for epoch in range(num_epochs):
                MODEL.train()
                epoch_loss = 0.0
                for batch_idx, (inputs, targets) in enumerate(train_loader):
                    optimizer.zero_grad()
                    outputs = MODEL(inputs)
                    loss = criterion(outputs, targets)
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()

                train_losses.append(epoch_loss / len(train_loader))

                # Evaluate on validation set every epoch
                if epoch % 5 == 0:
                    MODEL.eval()
                    val_loss = 0.0
                    val_preds = []
                    val_true = []
                    
                    with torch.inference_mode():
                        for inputs, targets in val_loader:
                            outputs = MODEL(inputs)
                            loss = criterion(outputs, targets)
                            val_loss += loss.item()
                            
                            preds = torch.argmax(outputs, dim=1).cpu().numpy()
                            val_preds.extend(preds)
                            val_true.extend(targets.cpu().numpy())
                    
                    val_losses.append(val_loss / len(val_loader))
                    
                    # Calculate F1 score
                    val_f1 = f1_score(val_true, val_preds, average='macro', zero_division=0)
                    val_f1_scores.append(val_f1)
                    
                    # Count predictions per class
                    unique, counts = np.unique(val_preds, return_counts=True)
                    pred_dist = dict(zip(unique, counts))
                    
                    print(f"Epoch [{epoch+1}/{num_epochs}], "
                          f"Train Loss: {train_losses[-1]:.4f}, "
                          f"Val Loss: {val_losses[-1]:.4f}, "
                          f"Val F1: {val_f1:.4f}, "
                          f"Preds: {pred_dist}")

                    # Early stopping based on F1 score
                    if val_f1 > best_val_f1:
                        best_val_f1 = val_f1
                        epochs_no_improve = 0
                        best_model_state = MODEL.state_dict()
                    else:
                        epochs_no_improve += 1

                    if epochs_no_improve >= early_stop_patience:
                        print(f"Early stopping at epoch {epoch + 1}")
                        break

            # Load best model
            if best_model_state is not None:
                MODEL.load_state_dict(best_model_state)

            # Plot Loss Graph
            plt.figure(figsize=(10, 6))
            plt.plot(range(1, len(train_losses) + 1), train_losses, label='Train Loss')
            plt.plot(range(5, 5 * len(val_losses) + 1, 5), val_losses, label='Validation Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Training and Validation Loss')
            plt.legend()
            plt.grid(True)
            os.makedirs('plots', exist_ok=True)
            output_plot_path = f"plots/{' '.join(map(str, hidden_sizes))}.png"
            plt.savefig(output_plot_path, dpi=300, bbox_inches='tight')
            plt.close()

            # Evaluation
            MODEL.eval()
            with torch.inference_mode():
                train_outputs = []
                train_targets = []
                for inputs, targets in train_loader:
                    outputs = MODEL(inputs).cpu().numpy()
                    train_outputs.append(outputs)
                    train_targets.append(targets.cpu().numpy())
                train_outputs = np.concatenate(train_outputs)
                train_targets = np.concatenate(train_targets)

                val_outputs = []
                val_targets = []
                for inputs, targets in val_loader:
                    outputs = MODEL(inputs).cpu().numpy()
                    val_outputs.append(outputs)
                    val_targets.append(targets.cpu().numpy())
                val_outputs = np.concatenate(val_outputs)
                val_targets = np.concatenate(val_targets)

                test_outputs = []
                test_targets = []
                for inputs, targets in test_loader:
                    outputs = MODEL(inputs).cpu().numpy()
                    test_outputs.append(outputs)
                    test_targets.append(targets.cpu().numpy())
                test_outputs = np.concatenate(test_outputs)
                test_targets = np.concatenate(test_targets)

            train_predictions = np.argmax(train_outputs, axis=1)
            val_predictions = np.argmax(val_outputs, axis=1)
            test_predictions = np.argmax(test_outputs, axis=1)

            # Plot Confusion Matrices
            plot_confusion_matrix(train_targets, train_predictions, 
                                "Confusion Matrix - Train", 
                                f"plots/cm_train_{idx_sim}.png")
            plot_confusion_matrix(val_targets, val_predictions, 
                                "Confusion Matrix - Validation", 
                                'plots/last_cf_matrix.png')
            plot_confusion_matrix(test_targets, test_predictions, 
                                "Confusion Matrix - Test", 
                                f"plots/cm_test_{idx_sim}.png")

            # Metrics
            acc_train, f1_train, prec_train, rec_train = compute_metrics(
                Y_train, train_predictions, split='Train'
            )
            acc_val, f1_val, prec_val, rec_val = compute_metrics(
                Y_val, val_predictions, split='Val'
            )
            acc_test, f1_test, prec_test, rec_test = compute_metrics(
                Y_test, test_predictions, split='Test'
            )

            # CM
            cm_val = confusion_matrix(Y_val, val_predictions)
            print("\nConfusion matrix (val):\n", cm_val)

            METRIX[idx_sim, :] = [
                acc_train, f1_train, prec_train, rec_train,
                acc_val, f1_val, prec_val, rec_val,
                acc_test, f1_test, prec_test, rec_test
            ]

            if f1_val > best_val_score:
                best_val_score = f1_val
                best_model = MODEL
                print(f"\n*** New best model ***")
                print(f"Hidden: {hidden_sizes}, Dropout: {dropout}, Val F1: {best_val_score:.4f}")

            idx_sim += 1

    # Save results
    sim_list_idx = range(0, Nsim)
    sim_list_hiddens = []
    sim_list_dropouts = []
    for hid in hidden_layers:
        for dropout in dropouts:
            sim_list_hiddens.append('_'.join(map(str, hid)))
            sim_list_dropouts.append(dropout)

    df = pd.DataFrame({
        'SIM': sim_list_idx,
        'Hid': sim_list_hiddens,
        'Drp': sim_list_dropouts,
        'acc_train': METRIX[:, 0],
        'f1_train': METRIX[:, 1],
        'prec_train': METRIX[:, 2],
        'rec_train': METRIX[:, 3],
        'acc_val': METRIX[:, 4],
        'f1_val': METRIX[:, 5],
        'prec_val': METRIX[:, 6],
        'rec_val': METRIX[:, 7],
        'acc_test': METRIX[:, 8],
        'f1_test': METRIX[:, 9],
        'prec_test': METRIX[:, 10],
        'rec_test': METRIX[:, 11],
    })

    csv_path = os.path.join(root_path, 'results')
    os.makedirs(csv_path, exist_ok=True)
    results_path = f'results/FCNN_configs.csv'
    
    if os.path.exists(results_path):
        df.to_csv(results_path, mode='a', header=False, index=False)
    else:
        df.to_csv(results_path, index=False)

    print(f"\n{'='*60}")
    print(f"Best Val F1: {best_val_score:.4f}")
    print(f"{'='*60}")