import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, precision_recall_fscore_support


def compute_metrics(Y, OUT, split, verbose=False):
    acc = accuracy_score(Y, OUT)
    f1, prec, rec = None, None, None
    precision_train, recall_train, f1_train, support_train = precision_recall_fscore_support(Y, OUT, average=None)
    
    print(f"{split} metrics per class:")
    for i, (p, r, f, s) in enumerate(zip(precision_train, recall_train, f1_train, support_train)):
        print(f"Class {i}: precision={p:.3f}, recall={r:.3f}, f1={f:.3f}, support={s}")

        if i == 1:
            f1 = f
            prec = p
            rec = r
    print(f"Overall accuracy: {acc:.3f}\n")
    return acc, f1, prec, rec 


def read_csv(file_path, labels_of_interest=[16], random_state=42):
    #
    # Load CSV
    df = pd.read_csv(file_path)
    #
    # --------- ORIGINAL LABEL DISTRIBUTION ----------
    print("\n=== Original class distribution ===")
    class_counts = df['Label'].value_counts().sort_index()
    class_percent = df['Label'].value_counts(normalize=True).sort_index() * 100
    print(pd.DataFrame({
        'count': class_counts,
        'percent (%)': class_percent.round(2)
    }))
    #
    # --------- BINARY LABEL MAPPING ----------
    df['binary_label'] = df['Label'].apply(lambda x: 1 if x in labels_of_interest else 0)
    #
    # --------- BINARY DISTRIBUTION ----------
    bin_counts = df['binary_label'].value_counts().sort_index()
    bin_percent = df['binary_label'].value_counts(normalize=True).sort_index() * 100
    print(pd.DataFrame({
        'count': bin_counts,
        'percent (%)': bin_percent.round(2)
    }))
    
    # Split features / labels
    columns = []
    to_be_removed = ['Label', 'binary_label', 'flow_id', 'flow_id_init', 'file']
    for c in to_be_removed:
        if c in df.columns:
            columns.append(c)
    X = df.drop(columns=columns)
    y = df['binary_label']
    #
    # Stratified split
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=random_state, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=random_state, stratify=y_temp
    )
    #
    # --------- SPLIT DISTRIBUTIONS ----------
    print("\n=== Split distributions (binary) ===")
    for name, labels in zip(
        ['Train', 'Val', 'Test'],
        [y_train, y_val, y_test]
    ):
        counts = labels.value_counts().sort_index()
        perc = labels.value_counts(normalize=True).sort_index() * 100
        print(f"\n{name}:")
        print(pd.DataFrame({
            'count': counts,
            'percent (%)': perc.round(2)
        }))

    # Min-Max normalization on train
    scaler = MinMaxScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns
    )
    #
    # Apply transform on val and test
    X_val_scaled = pd.DataFrame(
        scaler.transform(X_val),
        columns=X_val.columns
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns
    )
    # Replace invalid data with -1
    X_train_scaled = X_train_scaled.fillna(-1)
    X_val_scaled   = X_val_scaled.fillna(-1)
    X_test_scaled  = X_test_scaled.fillna(-1)
    #
    # print("\nDone reading csv\n")
    #
    return X_train_scaled, y_train, X_val_scaled, y_val, X_test_scaled, y_test
