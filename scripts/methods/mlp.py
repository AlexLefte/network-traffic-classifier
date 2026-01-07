import numpy as np
import pandas as pd
import pickle
import argparse
import os
import matplotlib.pyplot as plt
#
import torch
import torch.nn       as nn
import torch.optim    as optim
from torch.utils.data import DataLoader, TensorDataset
#
from utils                   import read_csv, compute_metrics
from sklearn.utils           import shuffle
from sklearn.metrics         import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics         import accuracy_score, f1_score
#
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
            layers.append(nn.Dropout(dropout_rate))  # Add Dropout
            prev_size = size
        layers.append(nn.Linear(prev_size, output_size))
        layers.append(nn.Softmax(dim=1))  # For classification probabilities
        self.model = nn.Sequential(*layers)
    #
    def forward(self, x):
        return self.model(x)
#
#
def plot_confusion_matrix(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues)
    plt.title(title)
    plt.savefig('plots/last_cf_matrix.png')
#
#
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_file", type=str, required=True)
    args = parser.parse_args()
    #
    # Datasets
    root_path = ""
    file_path = args.csv_file
    #
    # Choose device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Running on: ' + str(device))
    #
    # Save best model
    best_model = None
    best_val_score = -float('inf')
    #
    # Configurations
    hidden_layers = [[128, 128, 128, 64, 32, 16], [32, 32, 16], [32, 32, 16], [32, 16], [128, 64], [64, 32, 32]]
    dropouts = [0.3, 0.5]
    output_size = 2
    batch_size = 256
    num_epochs = 100
    learning_rate = 0.001
    input_size = 29
    #
    Nsim = len(hidden_layers)*len(dropouts)
    idx_sim = 0
    METRIX_ = np.zeros((Nsim, 4))
    #
    # Read and shuffle data
    X_train_split, Y_train, X_val_split, Y_val, X_test_split, Y_test = read_csv(file_path, labels_of_interest=[14, 15, 16, 17])
    X_train_split, Y_train = shuffle(X_train_split, Y_train, random_state=42)
    #
    for hidden_sizes in hidden_layers:
        for dropout in dropouts:
            METRIX = []
            #
            # Define the tensors
            X_train_tensor = torch.tensor(np.array(X_train_split), dtype=torch.float32).to(device)
            Y_train_tensor = torch.tensor(np.array(Y_train), dtype=torch.long).to(device)
            X_val_tensor   = torch.tensor(np.array(X_val_split), dtype=torch.float32).to(device)
            Y_val_tensor   = torch.tensor(np.array(Y_val), dtype=torch.long).to(device)
            X_test_tensor  = torch.tensor(np.array(X_test_split), dtype=torch.float32).to(device)
            Y_test_tensor  = torch.tensor(np.array(Y_test), dtype=torch.long).to(device)
            #
            # Create DataLoaders
            train_dataset = TensorDataset(X_train_tensor, Y_train_tensor)
            val_dataset   = TensorDataset(X_val_tensor, Y_val_tensor)
            test_dataset  = TensorDataset(X_test_tensor, Y_test_tensor)
            #
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
            test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
            #
            # Model, Loss, Optimizer
            MODEL = MLP(input_size, hidden_sizes, output_size, dropout).to(device)
            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(MODEL.parameters(), lr=0.0001, weight_decay=1e-2)
            #
            # Training Loop with Loss Tracking
            train_losses = []
            val_losses   = []
            test_losses  = []
            #
            # Early stopping parameters
            early_stop_patience = 10  # Number of epochs to wait for improvement
            best_val_loss = float('inf')
            epochs_no_improve = 0
            #
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
                #
                train_losses.append(epoch_loss / len(train_loader))
                #
                # Evaluate on validation set every few epochs
                if (epoch) % 5 == 0:
                    MODEL.eval()
                    val_loss = 0.0
                    with torch.inference_mode():
                        for inputs, targets in val_loader:
                            outputs = MODEL(inputs)
                            loss = criterion(outputs, targets)
                            val_loss += loss.item()
                    val_losses.append(val_loss / len(val_loader))
                    #
                    print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_losses[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}")
                    #
                    # Check early stopping condition
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        epochs_no_improve = 0
                        # Optionally save the best model
                        best_model = MODEL.state_dict()
                    else:
                        epochs_no_improve += 1
                    #
                    if epochs_no_improve >= early_stop_patience:
                        print(f"Early stopping triggered at epoch {epoch + 1}")
                        break
            #                
            # Plot Loss Graph
            plt.figure(figsize=(10, 6))
            plt.plot(range(1, len(train_losses) + 1), train_losses, label='Train Loss')
            plt.plot(range(10, 10 * len(val_losses) + 1, 10), val_losses, label='Validation Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Training and Validation Loss')
            plt.legend()
            plt.grid(True)
            os.makedirs('plots', exist_ok=True)
            output_plot_path = f"plots/{' '.join(map(str, hidden_sizes))}.png"  # Change to your desired path
            plt.savefig(output_plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            #
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
                #
                val_outputs = []
                val_targets = []
                for inputs, targets in val_loader:
                    outputs = MODEL(inputs).cpu().numpy()
                    val_outputs.append(outputs)
                    val_targets.append(targets.cpu().numpy())
                val_outputs = np.concatenate(val_outputs)
                val_targets = np.concatenate(val_targets)
                #
                test_outputs = []
                test_targets = []
                for inputs, targets in test_loader:
                    outputs = MODEL(inputs).cpu().numpy()
                    test_outputs.append(outputs)
                    test_targets.append(targets.cpu().numpy())
                test_outputs = np.concatenate(test_outputs)
                test_targets = np.concatenate(test_targets)
                #
                train_predictions = np.argmax(train_outputs, axis=1)
                val_predictions   = np.argmax(val_outputs, axis=1)
                test_predictions  = np.argmax(test_outputs, axis=1)
                #
                # Plot Confusion Matrices
                plot_confusion_matrix(train_targets, train_predictions, "Confusion Matrix - Train")
                plot_confusion_matrix(val_targets, val_predictions, "Confusion Matrix - Validation")
                plot_confusion_matrix(test_targets, test_predictions, "Confusion Matrix - Test")
            #    
            # Metrics
            acc_train, f1_train, *_ = compute_metrics(Y_train, train_predictions, split='Train')
            params_string = '_'.join(list(map(str, [hidden_sizes] + [dropout])))
            print(f'\n{params_string}')
            print(f'acc (train) = {acc_train}. f1 (train) = {f1_train}')
            #
            acc_val, f1_val, *_ = compute_metrics(Y_val, val_predictions, split='Val')
            print(f'acc (val) = {acc_val}. f1 (val) = {f1_val}')
            #
            acc_test, f1_test, *_ = compute_metrics(Y_test, test_predictions, split='Test')
            print(f'acc (test) = {acc_test}. f1 (test) = {f1_test}')
            METRIX += [acc_train, f1_train, acc_val, f1_val, acc_test, f1_test]
            #
            if f1_val > best_val_score:
                best_val_score = f1_val
                best_model = MODEL
                print(f"New best model found with hidden_layers: {' '.join(map(str, hidden_sizes))}, Dropout: {dropout}, Val Mean UA: {best_val_score:.2f}")
                #    
                # Save best model
                with open(f"models/best_mlp_{'_'.join(map(str, hidden_sizes))}_{dropout}_{acc_val}_{f1_val}.pkl", "wb") as f:
                    pickle.dump(best_model, f)
            #    
            idx_sim += 1
    #
    # Save to csv
    sim_list_idx = range(0, Nsim)
    sim_list_hiddens = []
    sim_list_dropouts = []
    for hid in hidden_layers:
        for dropout in dropouts:
            sim_list_hiddens.append('_'.join(map(str, hid)))
            sim_list_dropouts.append(dropout)
    #
    df_dict = { k:v for (k, v) in zip(['SIM', 'Hid', 'Drp',
                                        'Acc_train [%]', 'F1_train [%]',
                                        'Acc_val [%]', 'F1_val [%]'],
                                        [sim_list_idx,
                                        sim_list_hiddens,
                                        sim_list_dropouts,
                                        METRIX_[:,0], METRIX_[:,1],
                                        METRIX_[:,2], METRIX_[:,3]]) }
    df = pd.DataFrame(df_dict)
    results_path = f'FCNN_configs.csv'
    #
    if os.path.exists(results_path):
        df.to_csv(results_path, mode='a', header=False, index=False)
    else:
        df.to_csv(results_path, index=False)