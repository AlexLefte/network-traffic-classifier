import numpy as np
from scapy.all import rdpcap, IP, TCP, UDP
from scipy import fft, signal
from scipy.stats import skew, kurtosis
from collections import defaultdict
import pandas as pd

def extract_flow_key(pkt):
    """Extract 5-tuple flow identifier"""
    if IP in pkt:
        src_ip = pkt[IP].src
        dst_ip = pkt[IP].dst
        proto = pkt[IP].proto
        
        src_port = pkt.sport if hasattr(pkt, 'sport') else 0
        dst_port = pkt.dport if hasattr(pkt, 'dport') else 0
        
        # Normalize flow (bidirectional)
        if src_ip < dst_ip:
            return (src_ip, dst_ip, src_port, dst_port, proto)
        else:
            return (dst_ip, src_ip, dst_port, src_port, proto)
    return None

def extract_frequency_features(pcap_file, flow_duration=60):
    """
    Extract frequency domain features from PCAP file
    
    Parameters:
    - pcap_file: path to PCAP file
    - flow_duration: time window in seconds to aggregate flows
    
    Returns:
    - DataFrame with frequency domain features per flow
    """
    
    packets = rdpcap(pcap_file)
    flows = defaultdict(lambda: {
        'timestamps': [],
        'sizes': [],
        'iat': [],  # Inter-arrival times
        'start_time': None
    })
    
    print(f"Processing {len(packets)} packets...")
    
    # Group packets by flow
    for pkt in packets:
        flow_key = extract_flow_key(pkt)
        if flow_key and IP in pkt:
            timestamp = float(pkt.time)
            size = len(pkt)
            
            flow = flows[flow_key]
            
            if flow['start_time'] is None:
                flow['start_time'] = timestamp
            
            flow['timestamps'].append(timestamp)
            flow['sizes'].append(size)
            
            # Calculate inter-arrival time
            if len(flow['timestamps']) > 1:
                iat = timestamp - flow['timestamps'][-2]
                flow['iat'].append(iat)
    
    # Extract features for each flow
    features_list = []
    
    for flow_key, flow_data in flows.items():
        if len(flow_data['iat']) < 10:  # Skip flows with too few packets
            continue
        
        iat = np.array(flow_data['iat'])
        sizes = np.array(flow_data['sizes'])
        
        # === FREQUENCY DOMAIN FEATURES ===
        
        # 1. FFT of Inter-Arrival Times
        iat_fft = np.abs(fft.fft(iat))
        iat_freq = fft.fftfreq(len(iat), d=np.mean(iat))
        
        # FFT statistics
        fft_mean = np.mean(iat_fft)
        fft_std = np.std(iat_fft)
        fft_max = np.max(iat_fft)
        fft_energy = np.sum(iat_fft ** 2)
        
        # 2. Power Spectral Density (PSD)
        freqs, psd = signal.periodogram(iat, fs=1/np.mean(iat))
        psd_mean = np.mean(psd)
        psd_std = np.std(psd)
        psd_max = np.max(psd)
        psd_energy = np.sum(psd)
        
        # Dominant frequency
        dominant_freq_idx = np.argmax(psd[1:]) + 1  # Skip DC component
        dominant_freq = freqs[dominant_freq_idx]
        dominant_power = psd[dominant_freq_idx]
        
        # 3. Spectral Entropy (regularity measure)
        psd_norm = psd / np.sum(psd)
        spectral_entropy = -np.sum(psd_norm * np.log2(psd_norm + 1e-12))
        
        # 4. Autocorrelation in frequency domain
        autocorr = np.correlate(iat - np.mean(iat), iat - np.mean(iat), mode='full')
        autocorr = autocorr[len(autocorr)//2:]
        autocorr_fft = np.abs(fft.fft(autocorr))
        autocorr_energy = np.sum(autocorr_fft ** 2)
        
        # 5. Welch's method for smoother PSD estimate
        freqs_welch, psd_welch = signal.welch(iat, fs=1/np.mean(iat), nperseg=min(256, len(iat)))
        psd_welch_mean = np.mean(psd_welch)
        psd_welch_max = np.max(psd_welch)
        
        # 6. Packet size FFT features
        size_fft = np.abs(fft.fft(sizes))
        size_fft_mean = np.mean(size_fft)
        size_fft_std = np.std(size_fft)
        
        # 7. Coefficient of Variation in frequency domain
        cv_freq = fft_std / (fft_mean + 1e-12)
        
        # 8. JITTER - Variation in inter-arrival times
        jitter_mean = np.mean(np.abs(np.diff(iat)))
        jitter_std = np.std(np.abs(np.diff(iat)))
        jitter_max = np.max(np.abs(np.diff(iat)))
        
        # RFC 3550 Jitter (weighted average)
        rfc_jitter = 0
        for i in range(1, len(iat)):
            rfc_jitter += (np.abs(iat[i] - iat[i-1]) - rfc_jitter) / 16.0
        
        # 9. SHIMMER - Variation in packet sizes (amplitude variation)
        shimmer_mean = np.mean(np.abs(np.diff(sizes)))
        shimmer_std = np.std(np.abs(np.diff(sizes)))
        shimmer_relative = shimmer_mean / (avg_packet_size + 1e-12)
        
        # Shimmer in dB (common in audio analysis)
        sizes_norm = sizes / (np.mean(sizes) + 1e-12)
        shimmer_db = 20 * np.log10(np.mean(np.abs(np.diff(sizes_norm))) + 1e-12)
        
        # 10. SKEWNESS - Asymmetry of distributions
        iat_skewness = skew(iat)
        size_skewness = skew(sizes)
        fft_skewness = skew(iat_fft)
        psd_skewness = skew(psd)
        
        # 11. KURTOSIS - Tail heaviness / peakedness
        iat_kurtosis = kurtosis(iat)
        size_kurtosis = kurtosis(sizes)
        fft_kurtosis = kurtosis(iat_fft)
        psd_kurtosis = kurtosis(psd)
        
        # === TIME DOMAIN FEATURES (for comparison) ===
        iat_mean = np.mean(iat)
        iat_std = np.std(iat)
        iat_cv = iat_std / (iat_mean + 1e-12)
        
        packet_rate = len(flow_data['timestamps']) / (flow_data['timestamps'][-1] - flow_data['timestamps'][0])
        avg_packet_size = np.mean(sizes)
        
        features = {
            'flow_key': str(flow_key),
            'num_packets': len(flow_data['timestamps']),
            
            # Frequency domain features
            'fft_mean': fft_mean,
            'fft_std': fft_std,
            'fft_max': fft_max,
            'fft_energy': fft_energy,
            'psd_mean': psd_mean,
            'psd_std': psd_std,
            'psd_max': psd_max,
            'psd_energy': psd_energy,
            'dominant_freq': dominant_freq,
            'dominant_power': dominant_power,
            'spectral_entropy': spectral_entropy,
            'autocorr_energy': autocorr_energy,
            'psd_welch_mean': psd_welch_mean,
            'psd_welch_max': psd_welch_max,
            'size_fft_mean': size_fft_mean,
            'size_fft_std': size_fft_std,
            'cv_freq': cv_freq,
            
            # Jitter features
            'jitter_mean': jitter_mean,
            'jitter_std': jitter_std,
            'jitter_max': jitter_max,
            'rfc_jitter': rfc_jitter,
            
            # Shimmer features
            'shimmer_mean': shimmer_mean,
            'shimmer_std': shimmer_std,
            'shimmer_relative': shimmer_relative,
            'shimmer_db': shimmer_db,
            
            # Skewness features
            'iat_skewness': iat_skewness,
            'size_skewness': size_skewness,
            'fft_skewness': fft_skewness,
            'psd_skewness': psd_skewness,
            
            # Kurtosis features
            'iat_kurtosis': iat_kurtosis,
            'size_kurtosis': size_kurtosis,
            'fft_kurtosis': fft_kurtosis,
            'psd_kurtosis': psd_kurtosis,
            
            # Time domain (for reference)
            'iat_mean': iat_mean,
            'iat_std': iat_std,
            'iat_cv': iat_cv,
            'packet_rate': packet_rate,
            'avg_packet_size': avg_packet_size,
        }
        
        features_list.append(features)
    
    df = pd.DataFrame(features_list)
    print(f"\nExtracted features for {len(df)} flows")
    return df

# Example usage
if __name__ == "__main__":
    pcap_file = "your_capture.pcap"  # Replace with your PCAP file
    
    # Extract features
    features_df = extract_frequency_features(pcap_file)
    
    # Save to CSV
    features_df.to_csv("voip_frequency_features.csv", index=False)
    
    # Display summary statistics
    print("\n=== Feature Statistics ===")
    print(features_df.describe())
    
    # Display first few flows
    print("\n=== Sample Features ===")
    print(features_df.head())