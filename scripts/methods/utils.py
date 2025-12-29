import pandas as pd
# 
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
#
# Helper function
def read_csv(filepath, id_column='ID', test_size=0.2, random_state=42):
    #
    # Load CSV
    df = pd.read_csv("flows.csv")
    #
    # Map labels to binary 
    labels_of_interest = [14]  # skype_voice
    df['binary_label'] = df['Label'].apply(lambda x: 1 if x in labels_of_interest else 0)
    #
    # Split features / labels 
    X = df.drop(columns=['Label', 'binary_label', 'flow_id'])  # excluzând ID-ul și labelul original
    y = df['binary_label']
    #
    # Stratified split 
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )
    #   
    # Min-Max normalization on train
    scaler = MinMaxScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
    #
    # Apply transform on val and test
    X_val_scaled = pd.DataFrame(scaler.transform(X_val), columns=X_val.columns)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)
    #
    #  Replace invalid data with -1 
    X_train_scaled = X_train_scaled.fillna(-1)
    X_val_scaled = X_val_scaled.fillna(-1)
    X_test_scaled = X_test_scaled.fillna(-1)
    #
    print("Done reading csv")
    #
    return X_train_scaled, y_train, X_val_scaled, y_val, X_test_scaled, y_test



