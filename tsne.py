import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import pandas as pd
import seaborn as sns

# Define the category mappings
CHAT = [0, 1, 21, 2, 3, 4]  # aim, facebook_chat, gmail, hangouts_chat, skype_chat, icq
EMAIL = [5]  # email
FILE_TRANSFER = [6, 7, 8, 9]  # ftps, scp, sftp, skype_file
STREAMING = [10, 11, 12, 13]  # netflix, spotify, vimeo, youtube
VOIP = [14, 15, 16, 17]  # facebook_audio, hangouts_audio, skype_audio, voipbuster
VIDEO_CALL = [18, 19, 20]  # facebook_video, skype_video, hangouts_video

def map_label_to_category(label):
    """
    Map numeric label to category name.
    """
    if label in CHAT:
        return 'CHAT'
    elif label in EMAIL:
        return 'EMAIL'
    elif label in FILE_TRANSFER:
        return 'FILE_TRANSFER'
    elif label in STREAMING:
        return 'STREAMING'
    elif label in VOIP:
        return 'VOIP'
    elif label in VIDEO_CALL:
        return 'VIDEO_CALL'
    else:
        return 'UNKNOWN'

def load_and_prepare_data(csv_path):
    """
    Load CSV data and prepare for t-SNE analysis.
    
    Args:
        csv_path: Path to CSV file
        
    Returns:
        df: Full DataFrame
        X: Feature matrix for t-SNE
        categories: Category labels (CHAT, EMAIL, etc.)
        labels: Original traffic type labels
    """
    # Load CSV
    df = pd.read_csv(csv_path)
    
    print(f"Loaded {len(df)} flows from {csv_path}")
    print(f"Columns: {df.columns.tolist()}")
    
    # Feature columns (all except flow_id and Label)
    feature_cols = [col for col in df.columns 
                   if col not in ['flow_id', 'Label']]
    
    print(f"\nUsing {len(feature_cols)} features for t-SNE:")
    print(feature_cols)
    
    # Extract features
    X = df[feature_cols].values
    
    # Handle NaN and inf values
    X = np.nan_to_num(X, nan=0.0, posinf=1e10, neginf=-1e10)
    
    # Extract labels
    labels = df['Label'].values
    
    # Map labels to categories
    categories = np.array([map_label_to_category(label) for label in labels])
    
    print(f"\nUnique labels: {np.unique(labels)}")
    print(f"Unique categories: {np.unique(categories)}")
    print(f"\nCategory distribution:")
    print(pd.Series(categories).value_counts())
    print(f"\nLabel distribution:")
    print(df['Label'].value_counts())
    
    return df, X, categories, labels, feature_cols

def plot_both_analyses(X, categories, labels, perplexity=30, n_iter=1000, random_state=42):
    """
    Create side-by-side plots for both category and label analysis.
    """
    # Standardize features once
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Run t-SNE once
    print(f"\nRunning t-SNE with {X.shape[0]} samples and {X.shape[1]} features...")
    print(f"Parameters: perplexity={perplexity}, n_iter={n_iter}")
    
    tsne = TSNE(n_components=2, perplexity=perplexity,
                random_state=random_state, verbose=1)
    X_tsne = tsne.fit_transform(X_scaled)
    
    # Create subplots
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    
    # Plot 1: By Category (6 groups)
    category_colors = {
        'CHAT': '#3b82f6',         # Blue
        'EMAIL': '#8b5cf6',        # Purple
        'FILE_TRANSFER': '#10b981', # Green
        'STREAMING': '#f59e0b',    # Orange
        'VOIP': '#ef4444',         # Red
        'VIDEO_CALL': '#ec4899',   # Pink
        'UNKNOWN': '#6b7280'       # Gray
    }
    
    unique_categories = np.unique(categories)
    
    for category in unique_categories:
        mask = categories == category
        count = np.sum(mask)
        axes[0].scatter(X_tsne[mask, 0], X_tsne[mask, 1], 
                       c=category_colors.get(category, '#6b7280'),
                       label=f'{category} (n={count})', 
                       alpha=0.6, s=50, edgecolors='white', linewidth=0.5)
    
    axes[0].set_xlabel('t-SNE Dimension 1', fontsize=12)
    axes[0].set_ylabel('t-SNE Dimension 2', fontsize=12)
    axes[0].set_title('t-SNE by Traffic Category (6 Groups)', fontsize=14, fontweight='bold')
    axes[0].legend(title='Category', fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: By Label (21 classes)
    unique_labels = np.unique(labels)
    n_classes = len(unique_labels)
    
    if n_classes <= 10:
        colors = sns.color_palette('tab10', n_colors=n_classes)
    else:
        colors = sns.color_palette('husl', n_colors=n_classes)
    
    label_colors = dict(zip(unique_labels, colors))
    
    for label in unique_labels:
        mask = labels == label
        count = np.sum(mask)
        axes[1].scatter(X_tsne[mask, 0], X_tsne[mask, 1], 
                       c=[label_colors[label]],
                       label=f'{label} (n={count})', 
                       alpha=0.6, s=50, edgecolors='white', linewidth=0.5)
    
    axes[1].set_xlabel('t-SNE Dimension 1', fontsize=12)
    axes[1].set_ylabel('t-SNE Dimension 2', fontsize=12)
    axes[1].set_title('t-SNE by Traffic Type (21 Classes)', fontsize=14, fontweight='bold')
    
    if n_classes > 10:
        axes[1].legend(title='Traffic Type', fontsize=8, ncol=2)
    else:
        axes[1].legend(title='Traffic Type', fontsize=10)
    
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    return X_tsne

def plot_tsne_by_category(X, categories, perplexity=30, n_iter=1000, random_state=42):
    """
    Plot t-SNE visualization colored by category (6 groups).
    """
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Run t-SNE
    print(f"\nRunning t-SNE with {X.shape[0]} samples and {X.shape[1]} features...")
    print(f"Parameters: perplexity={perplexity}, n_iter={n_iter}")
    
    tsne = TSNE(n_components=2, perplexity=perplexity,
                random_state=random_state, verbose=1)
    X_tsne = tsne.fit_transform(X_scaled)
    
    # Create plot
    plt.figure(figsize=(10, 8))
    
    # Define colors for categories
    category_colors = {
        'CHAT': '#3b82f6',         # Blue
        'EMAIL': '#8b5cf6',        # Purple
        'FILE_TRANSFER': '#10b981', # Green
        'STREAMING': '#f59e0b',    # Orange
        'VOIP': '#ef4444',         # Red
        'VIDEO_CALL': '#ec4899',   # Pink
        'UNKNOWN': '#6b7280'       # Gray
    }
    
    unique_categories = np.unique(categories)
    for category in unique_categories:
        mask = categories == category
        count = np.sum(mask)
        plt.scatter(X_tsne[mask, 0], X_tsne[mask, 1], 
                   c=category_colors.get(category, '#6b7280'),
                   label=f'{category} (n={count})', 
                   alpha=0.6, s=50, edgecolors='white', linewidth=0.5)
    
    plt.xlabel('t-SNE Dimension 1', fontsize=12)
    plt.ylabel('t-SNE Dimension 2', fontsize=12)
    plt.title('t-SNE Visualization by Traffic Category (6 Groups)', fontsize=14, fontweight='bold')
    plt.legend(title='Category', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    return X_tsne

def plot_tsne_by_label(X, labels, perplexity=30, n_iter=1000, random_state=42):
    """
    Plot t-SNE visualization colored by traffic label/class.
    """
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Run t-SNE
    print(f"\nRunning t-SNE with {X.shape[0]} samples and {X.shape[1]} features...")
    print(f"Parameters: perplexity={perplexity}, n_iter={n_iter}")
    
    tsne = TSNE(n_components=2, perplexity=perplexity,
                random_state=random_state, verbose=1)
    X_tsne = tsne.fit_transform(X_scaled)
    
    # Create plot
    plt.figure(figsize=(12, 8))
    
    # Use seaborn color palette for labels
    unique_labels = np.unique(labels)
    n_classes = len(unique_labels)
    
    # Choose appropriate color palette
    if n_classes <= 10:
        colors = sns.color_palette('tab10', n_colors=n_classes)
    else:
        colors = sns.color_palette('husl', n_colors=n_classes)
    
    label_colors = dict(zip(unique_labels, colors))
    
    for label in unique_labels:
        mask = labels == label
        count = np.sum(mask)
        plt.scatter(X_tsne[mask, 0], X_tsne[mask, 1], 
                   c=[label_colors[label]],
                   label=f'{label} (n={count})', 
                   alpha=0.6, s=50, edgecolors='white', linewidth=0.5)
    
    plt.xlabel('t-SNE Dimension 1', fontsize=12)
    plt.ylabel('t-SNE Dimension 2', fontsize=12)
    plt.title('t-SNE Visualization by Traffic Type (21 Classes)', fontsize=14, fontweight='bold')
    
    # Adjust legend position based on number of classes
    if n_classes > 10:
        plt.legend(title='Traffic Type', fontsize=8, bbox_to_anchor=(1.05, 1), 
                  loc='upper left', ncol=2)
    else:
        plt.legend(title='Traffic Type', fontsize=10, bbox_to_anchor=(1.05, 1), 
                  loc='upper left')
    
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    return X_tsne

def plot_both_analyses(X, protocols, labels, perplexity=30, n_iter=1000, random_state=42):
    """
    Create side-by-side plots for both protocol and label analysis.
    """
    # Standardize features once
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Run t-SNE once
    print(f"\nRunning t-SNE with {X.shape[0]} samples and {X.shape[1]} features...")
    print(f"Parameters: perplexity={perplexity}, n_iter={n_iter}")
    
    tsne = TSNE(n_components=2, perplexity=perplexity,
                random_state=random_state, verbose=1)
    X_tsne = tsne.fit_transform(X_scaled)
    
    # Create subplots
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    
    # Plot 1: By Protocol
    protocol_colors = {'TCP': '#3b82f6', 'UDP': '#ef4444'}
    unique_protocols = np.unique(protocols)
    
    for protocol in unique_protocols:
        mask = protocols == protocol
        count = np.sum(mask)
        axes[0].scatter(X_tsne[mask, 0], X_tsne[mask, 1], 
                       c=protocol_colors.get(protocol, '#6b7280'),
                       label=f'{protocol} (n={count})', 
                       alpha=0.6, s=50, edgecolors='white', linewidth=0.5)
    
    axes[0].set_xlabel('t-SNE Dimension 1', fontsize=12)
    axes[0].set_ylabel('t-SNE Dimension 2', fontsize=12)
    axes[0].set_title('t-SNE by Protocol (TCP/UDP)', fontsize=14, fontweight='bold')
    axes[0].legend(title='Protocol', fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: By Label
    unique_labels = np.unique(labels)
    n_classes = len(unique_labels)
    
    if n_classes <= 10:
        colors = sns.color_palette('tab10', n_colors=n_classes)
    else:
        colors = sns.color_palette('husl', n_colors=n_classes)
    
    label_colors = dict(zip(unique_labels, colors))
    
    for label in unique_labels:
        mask = labels == label
        count = np.sum(mask)
        axes[1].scatter(X_tsne[mask, 0], X_tsne[mask, 1], 
                       c=[label_colors[label]],
                       label=f'{label} (n={count})', 
                       alpha=0.6, s=50, edgecolors='white', linewidth=0.5)
    
    axes[1].set_xlabel('t-SNE Dimension 1', fontsize=12)
    axes[1].set_ylabel('t-SNE Dimension 2', fontsize=12)
    axes[1].set_title('t-SNE by Traffic Type (Label)', fontsize=14, fontweight='bold')
    
    if n_classes > 10:
        axes[1].legend(title='Traffic Type', fontsize=8, ncol=2)
    else:
        axes[1].legend(title='Traffic Type', fontsize=10)
    
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    return X_tsne

# Main execution
if __name__ == "__main__":
    # Path to your CSV file
    csv_path = r'C:\Users\cata\Documents\GitHub\network-traffic-classifier\scripts\methods\features_timeout_15s_2pkts_each_dir.csv'  # UPDATE THIS PATH
    
    # Load and prepare data
    df, X, categories, labels, feature_cols = load_and_prepare_data(csv_path)
    
    # Adjust perplexity based on dataset size
    # Rule of thumb: perplexity should be between 5 and 50, and less than n_samples
    n_samples = X.shape[0]
    perplexity = min(30, n_samples // 3) if n_samples < 100 else 30
    
    print(f"\nUsing perplexity: {perplexity}")
    
    # Option 1: Plot both analyses side by side (RECOMMENDED)
    print("\n" + "="*60)
    print("Creating combined visualization...")
    print("="*60)
    X_tsne = plot_both_analyses(X, categories, labels, perplexity=perplexity, n_iter=1000)
    plt.savefig('tsne_both_analyses.png', dpi=300, bbox_inches='tight')
    print("Saved: tsne_both_analyses.png")
    plt.show()
    
    # Option 2: Plot separately
    print("\n" + "="*60)
    print("Creating category visualization...")
    print("="*60)
    X_tsne_category = plot_tsne_by_category(X, categories, perplexity=perplexity)
    plt.savefig('tsne_category.png', dpi=300, bbox_inches='tight')
    print("Saved: tsne_category.png")
    plt.show()
    
    print("\n" + "="*60)
    print("Creating label/class visualization...")
    print("="*60)
    X_tsne_label = plot_tsne_by_label(X, labels, perplexity=perplexity)
    plt.savefig('tsne_label.png', dpi=300, bbox_inches='tight')
    print("Saved: tsne_label.png")
    plt.show()
    
    # Save results to CSV
    results_df = pd.DataFrame({
        'flow_id': df['flow_id'],
        'category': categories,
        'label': labels,
        'tsne_dim1': X_tsne[:, 0],
        'tsne_dim2': X_tsne[:, 1]
    })
    results_df.to_csv('tsne_results.csv', index=False)
    print("\nResults saved to tsne_results.csv")
    
    # Print summary statistics
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Total flows analyzed: {len(df)}")
    print(f"Number of features: {len(feature_cols)}")
    print(f"\nCategory distribution:")
    print(pd.Series(categories).value_counts())
    print(f"\nLabel distribution:")
    print(pd.Series(labels).value_counts())