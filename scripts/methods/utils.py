import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

def read_csv(file_path, id_column='ID', test_size=0.2, random_state=42):
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
    labels_of_interest = [14, 15, 16, 17] # 14, 15, 16, 17
    df['binary_label'] = df['Label'].apply(lambda x: 1 if x in labels_of_interest else 0)
    #
    # --------- BINARY DISTRIBUTION ----------
    # print("\n=== Binary class distribution ===")
    bin_counts = df['binary_label'].value_counts().sort_index()
    bin_percent = df['binary_label'].value_counts(normalize=True).sort_index() * 100
    print(pd.DataFrame({
        'count': bin_counts,
        'percent (%)': bin_percent.round(2)
    }))
    #
    # Split features / labels
    X = df.drop(columns=['Label', 'binary_label', 'flow_id'])
    # X = df.drop(columns=['Label', 'binary_label'])
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
    #
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
    #
    # Replace invalid data with -1
    X_train_scaled = X_train_scaled.fillna(-1)
    X_val_scaled   = X_val_scaled.fillna(-1)
    X_test_scaled  = X_test_scaled.fillna(-1)
    #
    # print("\nDone reading csv\n")
    #
    return X_train_scaled, y_train, X_val_scaled, y_val, X_test_scaled, y_test
