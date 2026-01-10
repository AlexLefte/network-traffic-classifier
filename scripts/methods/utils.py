import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, precision_recall_fscore_support
import numpy as np


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


def read_csv(file_path, labels_of_interest=[14,15,16,17], random_state=42):
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


def read_csv_multiclass(file_path, random_state=42):
    """
    Read CSV and prepare data for multi-class classification.
    Maps original labels to 6 categories: chat, email, file, streaming, voip, video_call
    Stratifies splits by flow_id to prevent data leakage.
    """
    # Category mappings
    CHAT = [0, 1, 21, 2, 3, 4]  # aim, facebook_chat, gmail, hangouts_chat, skype_chat, icq
    EMAIL = [5]  # email
    FILE = [6, 7, 8, 9]  # ftps, scp, sftp, skype_file
    STREAMING = [10, 11, 12, 13]  # netflix, spotify, vimeo, youtube
    VOIP = [14, 15, 16, 17]  # facebook_audio, hangouts_audio, skype_audio, voipbuster
    VIDEO_CALL = [18, 19, 20]  # facebook_video, skype_video, hangouts_video
    
    # Load CSV
    df = pd.read_csv(file_path)
    
    # --------- ORIGINAL LABEL DISTRIBUTION ----------
    print("\n=== Original class distribution ===")
    class_counts = df['Label'].value_counts().sort_index()
    class_percent = df['Label'].value_counts(normalize=True).sort_index() * 100
    print(pd.DataFrame({
        'count': class_counts,
        'percent (%)': class_percent.round(2)
    }))
    
    # --------- MULTI-CLASS LABEL MAPPING ----------
    def map_label_to_category(label):
        if label in CHAT:
            return 0  # chat
        elif label in EMAIL:
            return 1  # email
        elif label in FILE:
            return 2  # file
        elif label in STREAMING:
            return 3  # streaming
        elif label in VOIP:
            return 4  # voip
        elif label in VIDEO_CALL:
            return 5  # video_call
        else:
            return -1  # unknown
    
    df['multiclass_label'] = df['Label'].apply(map_label_to_category)
    
    # Category names for display
    category_names = {
        0: 'chat',
        1: 'email',
        2: 'file',
        3: 'streaming',
        4: 'voip',
        5: 'video_call'
    }
    
    # --------- MULTI-CLASS DISTRIBUTION ----------
    print("\n=== Multi-class distribution ===")
    mc_counts = df['multiclass_label'].value_counts().sort_index()
    mc_percent = df['multiclass_label'].value_counts(normalize=True).sort_index() * 100
    print(pd.DataFrame({
        'category': [category_names.get(i, 'unknown') for i in mc_counts.index],
        'count': mc_counts.values,
        'percent (%)': mc_percent.round(2).values
    }))
    
    # --------- FLOW-BASED STRATIFIED SPLIT ----------
    # Get unique flows with their labels
    flow_col = 'flow_id' if 'flow_id' in df.columns else 'flow_id_init'
    flow_labels = df.groupby(flow_col)['multiclass_label'].first()
    
    print(f"\n=== Flow-based split (using '{flow_col}') ===")
    print(f"Total unique flows: {len(flow_labels)}")
    
    # Split flows (not individual samples) - stratified by class
    flow_train, flow_temp, y_flow_train, y_flow_temp = train_test_split(
        flow_labels.index, 
        flow_labels.values,
        test_size=0.3, 
        random_state=random_state, 
        stratify=flow_labels.values
    )
    
    flow_val, flow_test, y_flow_val, y_flow_test = train_test_split(
        flow_temp, 
        y_flow_temp,
        test_size=0.5, 
        random_state=random_state, 
        stratify=y_flow_temp
    )
    
    # Create train/val/test sets based on flow assignments
    train_mask = df[flow_col].isin(flow_train)
    val_mask = df[flow_col].isin(flow_val)
    test_mask = df[flow_col].isin(flow_test)
    
    # Prepare features
    columns = []
    to_be_removed = ['Label', 'multiclass_label', 'binary_label', 'Category', 'flow_id', 'flow_id_init', 'file']
    for c in to_be_removed:
        if c in df.columns:
            columns.append(c)
    
    X = df.drop(columns=columns)
    y = df['multiclass_label']
    
    X_train = X[train_mask]
    y_train = y[train_mask]
    X_val = X[val_mask]
    y_val = y[val_mask]
    X_test = X[test_mask]
    y_test = y[test_mask]
    
    # --------- SPLIT DISTRIBUTIONS ----------
    print("\n=== Split distributions (multi-class) ===")
    for name, labels in zip(
        ['Train', 'Val', 'Test'],
        [y_train, y_val, y_test]
    ):
        counts = labels.value_counts().sort_index()
        perc = labels.value_counts(normalize=True).sort_index() * 100
        print(f"\n{name}: {len(labels)} samples")
        print(pd.DataFrame({
            'category': [category_names.get(i, 'unknown') for i in counts.index],
            'count': counts.values,
            'percent (%)': perc.round(2).values
        }))
    
    # Min-Max normalization on train
    scaler = MinMaxScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index
    )
    
    # Apply transform on val and test
    X_val_scaled = pd.DataFrame(
        scaler.transform(X_val),
        columns=X_val.columns,
        index=X_val.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns,
        index=X_test.index
    )
    
    # Replace invalid data with -1
    X_train_scaled = X_train_scaled.fillna(-1)
    X_val_scaled = X_val_scaled.fillna(-1)
    X_test_scaled = X_test_scaled.fillna(-1)
    
    # --------- CLASS WEIGHTS ----------
    # Calculate class weights for imbalanced classes
    from sklearn.utils.class_weight import compute_class_weight
    
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_train),
        y=y_train
    )
    class_weight_dict = dict(enumerate(class_weights))
    
    print("\n=== Class weights (balanced) ===")
    for cls, weight in class_weight_dict.items():
        print(f"{category_names.get(cls, 'unknown'):12s}: {weight:.4f}")
    
    print("\n=== Data preparation complete ===")
    print(f"Number of classes: {y.nunique()}")
    print(f"Feature dimensions: {X_train_scaled.shape[1]}")
    print(f"Split by flow_id to prevent data leakage")
    
    return X_train_scaled, y_train, X_val_scaled, y_val, X_test_scaled, y_test, class_weight_dict


from scipy.io import arff

def read_arff_multiclass(file_path, random_state=42):
    """
    Read ARFF and prepare data for multi-class classification.
    Maps ARFF labels to numeric categories: BROWSING, CHAT, STREAMING, MAIL, VOIP, P2P, FT
    """
    # Category mappings (string -> numeric)
    category_mapping = {
        'BROWSING': 0,
        'CHAT': 1,
        'STREAMING': 2,
        'MAIL': 3,
        'VOIP': 4,
        'P2P': 5,
        'FT': 6
    }
    
    # Reverse mapping for display
    category_names = {v: k for k, v in category_mapping.items()}
    
    # Load ARFF
    data, meta = arff.loadarff(file_path)
    df = pd.DataFrame(data)
    
    # ARFF stores categorical values as bytes, decode them
    # Assuming the class/label column is the last column or named 'class'
    label_column = df.columns[-1]  # Usually the last column in ARFF
    
    # Decode byte strings to regular strings
    if df[label_column].dtype == object:
        df[label_column] = df[label_column].str.decode('utf-8')
    
    # --------- ORIGINAL LABEL DISTRIBUTION ----------
    print("\n=== Original class distribution ===")
    class_counts = df[label_column].value_counts().sort_index()
    class_percent = df[label_column].value_counts(normalize=True).sort_index() * 100
    print(pd.DataFrame({
        'count': class_counts,
        'percent (%)': class_percent.round(2)
    }))
    
    # --------- MULTI-CLASS LABEL MAPPING ----------
    df['multiclass_label'] = df[label_column].map(category_mapping)
    
    # Check for unmapped labels
    unmapped = df[df['multiclass_label'].isna()]
    if len(unmapped) > 0:
        print(f"\nWarning: Found {len(unmapped)} rows with unmapped labels:")
        print(unmapped[label_column].unique())
    
    # --------- MULTI-CLASS DISTRIBUTION ----------
    print("\n=== Multi-class distribution ===")
    mc_counts = df['multiclass_label'].value_counts().sort_index()
    mc_percent = df['multiclass_label'].value_counts(normalize=True).sort_index() * 100
    print(pd.DataFrame({
        'category': [category_names.get(int(i), 'unknown') for i in mc_counts.index],
        'count': mc_counts.values,
        'percent (%)': mc_percent.round(2).values
    }))
    
    # Split features / labels
    # Remove the original label column and multiclass_label
    X = df.drop(columns=[label_column, 'multiclass_label'])
    y = df['multiclass_label']
    
    # Remove any remaining non-numeric columns
    non_numeric = X.select_dtypes(include=['object']).columns
    if len(non_numeric) > 0:
        print(f"\nRemoving non-numeric columns: {list(non_numeric)}")
        X = X.drop(columns=non_numeric)
    
    # Stratified split
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=random_state, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=random_state, stratify=y_temp
    )
    
    # --------- SPLIT DISTRIBUTIONS ----------
    print("\n=== Split distributions (multi-class) ===")
    for name, labels in zip(
        ['Train', 'Val', 'Test'],
        [y_train, y_val, y_test]
    ):
        counts = labels.value_counts().sort_index()
        perc = labels.value_counts(normalize=True).sort_index() * 100
        print(f"\n{name}:")
        print(pd.DataFrame({
            'category': [category_names.get(int(i), 'unknown') for i in counts.index],
            'count': counts.values,
            'percent (%)': perc.round(2).values
        }))
    
    # Min-Max normalization on train
    scaler = MinMaxScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns
    )
    
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
    X_val_scaled = X_val_scaled.fillna(-1)
    X_test_scaled = X_test_scaled.fillna(-1)
    
    print("\n=== Data preparation complete ===")
    print(f"Number of classes: {int(y.nunique())}")
    print(f"Feature dimensions: {X_train_scaled.shape[1]}")
    
    return X_train_scaled, y_train, X_val_scaled, y_val, X_test_scaled, y_test