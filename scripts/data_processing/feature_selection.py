# feature_analysis.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_classif

class FeatureAnalyzer:
    """
    Analizează importanța feature-urilor folosind corelația Pearson și PCA
    """
    
    def __init__(self, csv_path=None, df=None):
        """
        Args:
            csv_path: calea către fișierul CSV
            df: sau DataFrame direct
        """
        if csv_path:
            self.df = pd.read_csv(csv_path)
        elif df is not None:
            self.df = df.copy()
        else:
            raise ValueError("Trebuie să furnizezi fie csv_path fie df")
        
        print(f"✓ Date încărcate: {len(self.df)} fluxuri, {len(self.df.columns)} coloane")
        print(f"✓ Coloane: {list(self.df.columns)}")
        
        self.numeric_features = None
        self.correlation_matrix = None
        self.pca_model = None
        self.scaler = None
        self.has_label = 'Label' in self.df.columns
        
    def prepare_data(self, include_label=False, drop_nan=True):
        """
        Pregătește datele pentru analiză
        
        Args:
            include_label: include coloana Label
            drop_nan: elimină rândurile cu NaN (altfel le păstrează)
        """
        # Exclude coloanele non-numerice sau identificatori
        exclude_cols = ['flow_id']
        if not include_label and self.has_label:
            exclude_cols.append('Label')
        
        # Protocol poate fi numeric, îl includem
        self.numeric_features = [col for col in self.df.columns 
                                if col not in exclude_cols]
        
        # Selectează doar coloanele numerice
        X = self.df[self.numeric_features].copy()
        
        # Convertește la numeric și gestionează erorile
        for col in X.columns:
            X[col] = pd.to_numeric(X[col], errors='coerce')
        
        # Înlocuiește inf cu NaN (nu vrem inf în calcule)
        X.replace([np.inf, -np.inf], np.nan, inplace=True)
        
        # Afișează statistici despre valorile lipsă
        nan_counts = X.isna().sum()
        total_nans = nan_counts.sum()
        
        if total_nans > 0:
            print(f"\n⚠ Valori NaN/inf detectate:")
            for col, count in nan_counts[nan_counts > 0].items():
                pct = (count / len(X)) * 100
                print(f"  - {col}: {count} valori ({pct:.2f}%)")
            
            if drop_nan:
                rows_before = len(X)
                X = X.dropna()
                rows_after = len(X)
                rows_dropped = rows_before - rows_after
                print(f"\n  → Eliminate {rows_dropped} rânduri cu NaN ({rows_dropped/rows_before*100:.2f}%)")
                print(f"  → Rămân {rows_after} rânduri pentru analiză")
        
        return X
    
    def pearson_correlation_analysis(self, threshold=0.8):
        """
        Analizează corelațiile Pearson între feature-uri
        
        Args:
            threshold: pragul pentru identificarea feature-urilor corelate
            
        Returns:
            dict cu rezultatele analizei
        """
        X = self.prepare_data()
        
        # Calculează matricea de corelație
        self.correlation_matrix = X.corr(method='pearson')
        
        # Identifică perechi de feature-uri puternic corelate
        high_corr_pairs = []
        for i in range(len(self.correlation_matrix.columns)):
            for j in range(i+1, len(self.correlation_matrix.columns)):
                corr_val = self.correlation_matrix.iloc[i, j]
                if abs(corr_val) >= threshold:
                    high_corr_pairs.append({
                        'feature_1': self.correlation_matrix.columns[i],
                        'feature_2': self.correlation_matrix.columns[j],
                        'correlation': corr_val
                    })
        
        # Sortează după valoarea absolută a corelației
        high_corr_pairs = sorted(
            high_corr_pairs, 
            key=lambda x: abs(x['correlation']), 
            reverse=True
        )
        
        return {
            'correlation_matrix': self.correlation_matrix,
            'high_corr_pairs': high_corr_pairs,
            'num_high_corr': len(high_corr_pairs)
        }
    
    def pca_analysis(self, n_components=None, variance_threshold=0.95):
        """
        Efectuează analiza PCA
        
        Args:
            n_components: numărul de componente principale (None = toate)
            variance_threshold: pragul varianței cumulative
            
        Returns:
            dict cu rezultatele PCA
        """
        X = self.prepare_data()
        
        # Standardizare
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # PCA
        self.pca_model = PCA(n_components=n_components)
        X_pca = self.pca_model.fit_transform(X_scaled)
        
        # Calculează variența cumulativă
        cumulative_variance = np.cumsum(self.pca_model.explained_variance_ratio_)
        
        # Găsește numărul optim de componente
        n_components_optimal = np.argmax(cumulative_variance >= variance_threshold) + 1
        
        # Importanța feature-urilor pentru fiecare componentă principală
        components_df = pd.DataFrame(
            self.pca_model.components_,
            columns=X.columns,
            index=[f'PC{i+1}' for i in range(len(self.pca_model.components_))]
        )
        
        # Calculează importanța absolută a fiecărui feature (suma pe toate PC)
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': np.abs(self.pca_model.components_).sum(axis=0)
        }).sort_values('importance', ascending=False)
        
        # Contribuția fiecărui feature la primele N componente
        top_n = min(5, len(self.pca_model.components_))
        top_components_contribution = pd.DataFrame({
            'feature': X.columns,
            'contribution': np.abs(self.pca_model.components_[:top_n]).sum(axis=0)
        }).sort_values('contribution', ascending=False)
        
        # Loadings pentru primele 2 componente (pentru biplot)
        loadings = pd.DataFrame(
            self.pca_model.components_[:2].T,
            columns=['PC1', 'PC2'],
            index=X.columns
        )
        
        return {
            'pca_model': self.pca_model,
            'explained_variance_ratio': self.pca_model.explained_variance_ratio_,
            'cumulative_variance': cumulative_variance,
            'n_components_optimal': n_components_optimal,
            'components_df': components_df,
            'feature_importance': feature_importance,
            'top_components_contribution': top_components_contribution,
            'transformed_data': X_pca,
            'loadings': loadings
        }
    
    def mutual_information_analysis(self):
        """
        Calculează mutual information între features și Label
        (doar dacă există Label în dataset)
        """
        if not self.has_label:
            print("⚠ Nu există coloana 'Label' în dataset")
            return None
        
        # Pregătește features
        X = self.prepare_data(include_label=False, drop_nan=False)
        
        # Păstrează doar rândurile fără NaN
        valid_mask = ~X.isna().any(axis=1)
        X_clean = X[valid_mask]
        
        # Aplică aceeași mască pe Label
        y = self.df.loc[valid_mask, 'Label'].values
        
        print(f"  → Folosesc {len(X_clean)} rânduri valide pentru MI analysis")
        
        # Calculează mutual information
        mi_scores = mutual_info_classif(X_clean, y, random_state=42)
        
        mi_df = pd.DataFrame({
            'feature': X_clean.columns,
            'mi_score': mi_scores
        }).sort_values('mi_score', ascending=False)
        
        return mi_df
    
    def plot_correlation_heatmap(self, figsize=(16, 14), top_n=None, annot=False):
        """Plotează heatmap-ul corelațiilor"""
        if self.correlation_matrix is None:
            self.pearson_correlation_analysis()
        
        corr_matrix = self.correlation_matrix
        
        # Dacă vrem doar top N features
        if top_n:
            # Selectează top N după variabilitate
            variance = self.prepare_data().var().sort_values(ascending=False)
            top_features = variance.head(top_n).index.tolist()
            corr_matrix = self.correlation_matrix.loc[top_features, top_features]
        
        plt.figure(figsize=figsize)
        sns.heatmap(
            corr_matrix,
            cmap='coolwarm',
            center=0,
            annot=annot,
            fmt='.2f',
            square=True,
            linewidths=0.5,
            cbar_kws={'label': 'Pearson Correlation'}
        )
        plt.title('Pearson Correlation Matrix', fontsize=16, pad=20)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.show()
    
    def plot_pca_variance(self, pca_results):
        """Plotează variența explicată de componentele principale"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        n_components = len(pca_results['explained_variance_ratio'])
        
        # Variența explicată
        ax1.bar(
            range(1, n_components + 1),
            pca_results['explained_variance_ratio'],
            alpha=0.7,
            color='steelblue'
        )
        ax1.set_xlabel('Componentă Principală', fontsize=12)
        ax1.set_ylabel('Varianță Explicată', fontsize=12)
        ax1.set_title('Varianța Explicată de Fiecare PC', fontsize=14)
        ax1.grid(alpha=0.3, axis='y')
        
        # Afișează doar primele 20 componente dacă sunt prea multe
        if n_components > 20:
            ax1.set_xlim(0, 21)
        
        # Variența cumulativă
        ax2.plot(
            range(1, len(pca_results['cumulative_variance']) + 1),
            pca_results['cumulative_variance'],
            marker='o',
            color='coral',
            linewidth=2,
            markersize=4
        )
        ax2.axhline(y=0.95, color='red', linestyle='--', label='95% varianță', linewidth=2)
        ax2.axvline(
            x=pca_results['n_components_optimal'], 
            color='green', 
            linestyle='--', 
            label=f"Optimal: {pca_results['n_components_optimal']} PC",
            linewidth=2
        )
        ax2.set_xlabel('Număr Componente', fontsize=12)
        ax2.set_ylabel('Varianță Cumulativă', fontsize=12)
        ax2.set_title('Varianța Cumulativă', fontsize=14)
        ax2.legend(fontsize=10)
        ax2.grid(alpha=0.3)
        ax2.set_ylim([0, 1.05])
        
        plt.tight_layout()
        plt.show()
    
    def plot_feature_importance(self, pca_results, top_n=15):
        """Plotează importanța feature-urilor"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
        
        # Importanță totală
        top_features = pca_results['feature_importance'].head(top_n)
        colors1 = plt.cm.viridis(np.linspace(0.3, 0.9, len(top_features)))
        ax1.barh(range(len(top_features)), top_features['importance'], color=colors1)
        ax1.set_yticks(range(len(top_features)))
        ax1.set_yticklabels(top_features['feature'], fontsize=10)
        ax1.set_xlabel('Importanță (suma abs pe toate PC)', fontsize=11)
        ax1.set_title(f'Top {top_n} Feature-uri - Importanță Totală PCA', fontsize=13, fontweight='bold')
        ax1.invert_yaxis()
        ax1.grid(axis='x', alpha=0.3)
        
        # Contribuție la primele componente
        top_contrib = pca_results['top_components_contribution'].head(top_n)
        colors2 = plt.cm.plasma(np.linspace(0.3, 0.9, len(top_contrib)))
        ax2.barh(range(len(top_contrib)), top_contrib['contribution'], color=colors2)
        ax2.set_yticks(range(len(top_contrib)))
        ax2.set_yticklabels(top_contrib['feature'], fontsize=10)
        ax2.set_xlabel('Contribuție (primele 5 PC)', fontsize=11)
        ax2.set_title(f'Top {top_n} Feature-uri - Contribuție la Primele 5 PC', fontsize=13, fontweight='bold')
        ax2.invert_yaxis()
        ax2.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def plot_pca_biplot(self, pca_results, top_n_features=10):
        """Plotează biplot pentru primele 2 componente principale"""
        loadings = pca_results['loadings']
        
        # Selectează top N features după importanță
        top_features = pca_results['feature_importance'].head(top_n_features)['feature'].tolist()
        
        plt.figure(figsize=(12, 10))
        
        # Plotează vectorii pentru top features
        for feature in top_features:
            x, y = loadings.loc[feature, 'PC1'], loadings.loc[feature, 'PC2']
            plt.arrow(0, 0, x*3, y*3, head_width=0.05, head_length=0.05, 
                     fc='red', ec='red', alpha=0.6, linewidth=1.5)
            plt.text(x*3.2, y*3.2, feature, fontsize=9, ha='center', va='center',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.5))
        
        plt.xlabel(f'PC1 ({pca_results["explained_variance_ratio"][0]*100:.1f}%)', fontsize=12)
        plt.ylabel(f'PC2 ({pca_results["explained_variance_ratio"][1]*100:.1f}%)', fontsize=12)
        plt.title(f'PCA Biplot - Top {top_n_features} Features', fontsize=14, fontweight='bold')
        plt.axhline(y=0, color='k', linewidth=0.5, linestyle='--', alpha=0.3)
        plt.axvline(x=0, color='k', linewidth=0.5, linestyle='--', alpha=0.3)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def plot_mutual_information(self, mi_results, top_n=15):
        """Plotează mutual information cu Label"""
        if mi_results is None:
            return
        
        top_mi = mi_results.head(top_n)
        
        plt.figure(figsize=(12, 7))
        colors = plt.cm.coolwarm(np.linspace(0.2, 0.8, len(top_mi)))
        plt.barh(range(len(top_mi)), top_mi['mi_score'], color=colors)
        plt.yticks(range(len(top_mi)), top_mi['feature'], fontsize=10)
        plt.xlabel('Mutual Information Score', fontsize=12)
        plt.title(f'Top {top_n} Features - Mutual Information cu Label', 
                 fontsize=14, fontweight='bold')
        plt.gca().invert_yaxis()
        plt.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def generate_report(self, threshold=0.8, variance_threshold=0.95):
        """Generează un raport complet"""
        print("="*90)
        print(" "*25 + "ANALIZĂ FEATURE-URI - RAPORT COMPLET")
        print("="*90)
        
        # Info dataset
        print(f"\n📊 INFORMAȚII DATASET")
        print("-"*90)
        print(f"  Număr fluxuri: {len(self.df)}")
        print(f"  Număr features: {len([c for c in self.df.columns if c not in ['flow_id', 'Label']])}")
        if self.has_label:
            print(f"  Distribuție Label: {dict(self.df['Label'].value_counts())}")
        
        # Analiza Pearson
        print(f"\n\n1️⃣  ANALIZA CORELAȚIEI PEARSON")
        print("-"*90)
        pearson_results = self.pearson_correlation_analysis(threshold)
        
        print(f"\n  Feature-uri analizate: {len(self.numeric_features)}")
        print(f"  Perechi cu |corelație| > {threshold}: {pearson_results['num_high_corr']}")
        
        if pearson_results['high_corr_pairs']:
            print(f"\n  🔥 Top 10 perechi cu cea mai mare corelație:")
            for i, pair in enumerate(pearson_results['high_corr_pairs'][:10], 1):
                print(f"     {i:2d}. {pair['feature_1']:25s} <-> {pair['feature_2']:25s}: {pair['correlation']:7.4f}")
        else:
            print(f"  ✓ Nu există perechi cu corelație > {threshold}")
        
        # Analiza PCA
        print(f"\n\n2️⃣  ANALIZA PCA (Principal Component Analysis)")
        print("-"*90)
        pca_results = self.pca_analysis(variance_threshold=variance_threshold)
        
        reduction = (1 - pca_results['n_components_optimal']/len(self.numeric_features)) * 100
        print(f"\n  Componente optime (>{variance_threshold*100:.0f}% varianță): {pca_results['n_components_optimal']}")
        print(f"  Reducere dimensionalitate: {len(self.numeric_features)} → {pca_results['n_components_optimal']} ({reduction:.1f}% reducere)")
        
        print(f"\n  📈 Varianță explicată de primele 10 componente:")
        for i in range(min(10, len(pca_results['explained_variance_ratio']))):
            cum_var = pca_results['cumulative_variance'][i]
            print(f"     PC{i+1:2d}: {pca_results['explained_variance_ratio'][i]*100:6.2f}%  (cumulativ: {cum_var*100:6.2f}%)")
        
        print(f"\n  🌟 Top 15 cele mai importante feature-uri (PCA):")
        for idx, row in pca_results['feature_importance'].head(15).iterrows():
            print(f"     {idx+1:2d}. {row['feature']:30s}: {row['importance']:.6f}")
        
        # Mutual Information (dacă există Label)
        if self.has_label:
            print(f"\n\n3️⃣  ANALIZA MUTUAL INFORMATION (cu Label)")
            print("-"*90)
            mi_results = self.mutual_information_analysis()
            
            print(f"\n  🎯 Top 15 features cu cea mai mare MI cu Label:")
            for idx, row in mi_results.head(15).iterrows():
                print(f"     {idx+1:2d}. {row['feature']:30s}: {row['mi_score']:.6f}")
        
        # Recomandări
        print(f"\n\n4️⃣  RECOMANDĂRI")
        print("-"*90)
        
        # Feature-uri redundante
        very_high = [p for p in pearson_results['high_corr_pairs'] if abs(p['correlation']) > 0.95]
        high = [p for p in pearson_results['high_corr_pairs'] if 0.9 <= abs(p['correlation']) <= 0.95]
        
        if very_high:
            print(f"\n  ⚠️  Feature-uri FOARTE corelate (|r| > 0.95) - recomand ELIMINARE:")
            for pair in very_high[:8]:
                print(f"      → Elimină '{pair['feature_2']}' (păstrează '{pair['feature_1']}')")
        
        if high:
            print(f"\n  ⚡ Feature-uri corelate puternic (0.9 < |r| ≤ 0.95) - consideră eliminare:")
            for pair in high[:5]:
                print(f"      → Pair: '{pair['feature_1']}' <-> '{pair['feature_2']}' (r={pair['correlation']:.3f})")
        
        # Feature-uri importante de păstrat
        print(f"\n  ✅ Feature-uri ESENȚIALE (top 10 din PCA):")
        for feat in pca_results['feature_importance'].head(10)['feature'].tolist():
            print(f"      → {feat}")
        
        if self.has_label:
            print(f"\n  🎯 Feature-uri importante pentru clasificare (top 10 MI):")
            for feat in mi_results.head(10)['feature'].tolist():
                print(f"      → {feat}")
        
        print(f"\n  💡 Sugestie finală:")
        print(f"      Poți reduce de la {len(self.numeric_features)} la ~{pca_results['n_components_optimal']}")
        print(f"      feature-uri păstrând {variance_threshold*100:.0f}% din informație!")
        
        print("\n" + "="*90 + "\n")
        
        return {
            'pearson_results': pearson_results,
            'pca_results': pca_results,
            'mi_results': mi_results if self.has_label else None
        }


# ==================== UTILIZARE ====================
if __name__ == "__main__":
    # Înlocuiește cu calea către CSV-ul tău
    csv_path = r'C:\Users\cata\Documents\GitHub\network-traffic-classifier\scripts\methods\flows.csv'  # <-- MODIFICĂ AICI
    
    print("🚀 Pornesc analiza feature-urilor...\n")
    
    # Creează analizatorul
    analyzer = FeatureAnalyzer(csv_path=csv_path)
    
    # Generează raportul complet
    results = analyzer.generate_report(threshold=0.8, variance_threshold=0.95)
    
    # Generează toate vizualizările
    print("\n📊 Generez vizualizări...\n")
    
    print("1. Heatmap corelații Pearson...")
    analyzer.plot_correlation_heatmap(figsize=(18, 16), annot=False)
    
    print("2. Varianță explicată PCA...")
    analyzer.plot_pca_variance(results['pca_results'])
    
    print("3. Importanța feature-urilor...")
    analyzer.plot_feature_importance(results['pca_results'], top_n=20)
    
    print("4. PCA Biplot...")
    analyzer.plot_pca_biplot(results['pca_results'], top_n_features=12)
    
    if results['mi_results'] is not None:
        print("5. Mutual Information cu Label...")
        analyzer.plot_mutual_information(results['mi_results'], top_n=20)
    
    print("\n✅ Analiză completă!\n")