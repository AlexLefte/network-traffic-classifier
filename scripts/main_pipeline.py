# main_pipeline.py

import pandas as pd
import os
import sys
from typing import List, Dict, Any
from scapy.all import Ether, IP, TCP, wrpcap 
from sklearn.preprocessing import LabelEncoder # Adăugat pentru coerența finală a etichetei

# --- Configuratii ---
PCAP_FILE_NAME = "youtube2.pcap" 
OUTPUT_CSV_PATH = "youtube_features_analysis.csv"
FLOW_TIMEOUT = 600.0 # Timeout-ul ISCX [4, 5]

# Adaugă directorul 'src' la PATH
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Importă modulele
from src.pcap_ingest import pcap_reader_generator, normalize_flow_key
from src.flow_aggregator import aggregate_flows
from src.feature_extractor import extract_all_features, ExtractedFeatures

def simulate_pcap(filepath: str):
    """
    Creează un fișier PCAP simulat, modelând traficul în rafală (bursty) al YouTube.
    """
    start_time = 1678886400.0 

    # Flux 1: Trafic scurt
    p1 = Ether()/IP(src="192.168.1.10", dst="172.217.0.1")/TCP(sport=12345, dport=443, flags='S')
    p1.time = start_time + 0.0
    p2 = Ether()/IP(src="172.217.0.1", dst="192.168.1.10")/TCP(sport=443, dport=12345, flags='A')
    p2.time = start_time + 0.01

    # Flux 2: Trafic de streaming (Rafală de date B->A)
    p3 = Ether()/IP(src="192.168.1.10", dst="203.0.113.5")/TCP(sport=50000, dport=443, flags='A')
    p3.time = start_time + 1.0
    
    p4 = Ether()/IP(src="203.0.113.5", dst="192.168.1.10")/TCP(sport=443, dport=50000, flags='A')
    p4.time = start_time + 1.1 
    p4[IP].len = 1500 
    
    p5 = p4.copy() 
    p5.time = start_time + 1.15
    p5[IP].len = 1500 
    
    p6 = p4.copy() 
    p6.time = start_time + 1.25
    p6[IP].len = 800 
    
    p7 = Ether()/IP(src="192.168.1.10", dst="203.0.113.5")/TCP(sport=50000, dport=443, flags='A')
    p7.time = start_time + 10.0

    packets = [p1, p2, p3, p4, p5, p6, p7]
    wrpcap(filepath, packets)
    print(f"Fișier PCAP simulat creat la: {filepath} ({len(packets)} pachete)")


def run_pipeline(pcap_path: str, output_path: str, timeout: float):
    """
    Orchestrează întregul proces de extracție a feature-urilor în stil modular. [7]
    """
    if not os.path.exists(pcap_path):
        simulate_pcap(pcap_path)

    print(f"\nÎncepe procesarea în streaming a fișierului: {pcap_path}")

    # Pasul 1: Ingestie & Normalizare
    normalized_packets_generator = (
        (normalize_flow_key(pachet_dict), pachet_dict)
        for pachet_dict in pcap_reader_generator(pcap_path)
    )

    # Pasul 2: Agregare flux
    flows_generator = aggregate_flows(normalized_packets_generator, timeout=timeout)

    # Pasul 3: Extracția Feature-urilor (Flow_ID este transformat în hash numeric aici)
    features_generator = extract_all_features(flows_generator)
    
    final_features_list: List[ExtractedFeatures] = list(features_generator)

    if not final_features_list:
        print("Avertisment: Nu au fost extrase fluxuri.")
        return

    # Pasul 4: Salvarea rezultatelor
    df = pd.DataFrame(final_features_list)
    
    # 4.1. Codificarea Etichetei (pentru coerența coloanei Label)
    le = LabelEncoder()
    df['Label_Encoded'] = le.fit_transform(df['Label'])
    
    print("\nMaparea Etichetelor (Coerență Numerică):")
    label_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
    print(label_mapping)
    
    # 4.2. Reordonare și formatare pentru afișare/salvare ca tabel
    preferred_order = [
        'Flow_ID', 'Duration_s', 'Total_Packets', 'Bytes_AB', 'Bytes_BA',
        'Mean_Pkt_Len', 'STD_IAT', 'Kurtosis_IAT', 'Skewness_Pkt_Len',
        'Direction_Ratio', 'Label', 'Label_Encoded'
    ]
    cols_present = [c for c in preferred_order if c in df.columns]
    other_cols = [c for c in df.columns if c not in cols_present]
    df = df[cols_present + other_cols]

    # Round float columns to a readable number of decimals
    float_cols = df.select_dtypes(include=['float', 'float64', 'float32']).columns
    df[float_cols] = df[float_cols].round(6)

    # Save CSV with consistent float formatting
    df.to_csv(output_path, index=False, float_format='%.6f')

    # Also save a quick HTML table (optional, easy to open in browser)
    html_path = os.path.splitext(output_path)[0] + ".html"
    df.to_html(html_path, index=False)

    # Open the CSV automatically on Windows for quick inspection
    try:
        os.startfile(output_path)
    except OSError:
        # If running in an environment without a default app, ignore
        pass

    # Print a nicely formatted table to console
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 240)
    print("\nTabel rezultate (primele 20 rânduri):")
    print(df.head(20).to_string(index=False))

    print("-" * 50)
    print(f"Succes! Au fost extrase {len(final_features_list)} fluxuri.")
    print(f"Rezultatele au fost salvate în: {output_path}  și  {html_path}")
    print("-" * 50)


if __name__ == "__main__":
    run_pipeline(PCAP_FILE_NAME, OUTPUT_CSV_PATH, FLOW_TIMEOUT)