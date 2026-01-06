# src/feature_extractor.py

import numpy as np
from scipy.stats import kurtosis, skew
import hashlib

# Basic statistical functions
def compute_stats(arr):
    if len(arr) == 0:
        return 0, 0, -1, -1, 0
    return (
        np.mean(arr),
        np.median(arr),
        np.min(arr),
        np.max(arr),
        np.std(arr)
    )

# IAT-related functions
def compute_iat(packets):
            ts = sorted(p["timestamp"] for p in packets)
            return np.diff(ts) if len(ts) > 1 else np.array([]) 

# Main extraction function
def extract_features_from_flow(flow, file=None) -> dict:
    features = {}
    pkts = flow.get("packets", [])
    
    # Get initial direction
    init_ip, _ = flow["init_dir"]

    # Split packets by direction
    fwd_pkts, bwd_pkts = [], []
    for p in pkts:
        if p["ip_src"] == init_ip:
            fwd_pkts.append(p)
        else:
            bwd_pkts.append(p)

    # Get Duration
    duration = flow["duration"]
    if duration <= 0:
        duration = np.nan

    # Base features
    pkts_fwd = len(fwd_pkts)
    pkts_bwd = len(bwd_pkts)
    lens_fwd = [
        p.get("length", p.get("ip_len", p.get("pkt_len", 0)))
        for p in fwd_pkts
    ]
    lens_bwd = [
        p.get("length", p.get("ip_len", p.get("pkt_len", 0)))
        for p in bwd_pkts
    ]

    # Mean packet length-
    mean_pkt_len_fwd = np.mean(lens_fwd) if lens_fwd else 0
    mean_pkt_len_bwd = np.mean(lens_bwd) if lens_bwd else 0

    # Bytes/Packets Rates
    fb_psec = flow["total_bytes"] / duration if not np.isnan(duration) else np.nan
    fp_psec = flow["packet_count"] / duration if not np.isnan(duration) else np.nan

    # Fwd/Bwd Bytes/Packets Ratios
    # up_down_bytes_ratio = (
    #     sum(lens_fwd) / sum(lens_bwd)
    #     if sum(lens_bwd) > 0 else 0
    # )
    # up_down_pkt_ratio = (
    #     pkts_fwd / pkts_bwd
    #     if pkts_bwd > 0 else 0
    # )

    # IAT calculations
    fiat = compute_iat(fwd_pkts)
    biat = compute_iat(bwd_pkts)
    all_ts = sorted(p["timestamp"] for p in pkts)
    flowiat = np.diff(all_ts) if len(all_ts) > 1 else np.array([])

    # Statistical features
    fiat_total = np.sum(fiat) if len(fiat) > 0 else np.nan
    fiat_mean, fiat_median, fiat_min, fiat_max, fiat_std = compute_stats(fiat)

    biat_total = np.sum(biat) if len(biat) > 0 else np.nan
    biat_mean, biat_median, biat_min, biat_max, biat_std = compute_stats(biat)

    flowiat_mean, flowiat_median, flowiat_min, flowiat_max, flowiat_std = compute_stats(flowiat)

    flowiat_skew = skew(flowiat) if len(flowiat) > 2 else np.nan
    flowiat_kurt = kurtosis(flowiat) if len(flowiat) > 3 else np.nan

    sizes = np.array([p.get('length', p.get('ip_len', p.get('pkt_len', 0))) for p in pkts])
    size_kurtosis = kurtosis(sizes) if len(sizes) > 3 else np.nan
    size_skewness = skew(sizes) if len(sizes) > 2 else np.nan

    # Generate Flow ID as hash of flow key
    flow_id_string = "_".join(flow['flow_key'])
    flow_id_hash = int(hashlib.sha1(flow_id_string.encode('utf-8')).hexdigest(), 16) % (10**16)

    features = {
            'flow_id_init': flow_id_string,
            'file': file,
            'flow_id': flow_id_hash,
            'duration': flow['duration'],
            'protocol': flow['protocol'],

            # Fwd & Bwd Packet Counts
            "pkts_fwd": pkts_fwd,
            "pkts_bwd": pkts_bwd,

            # Mean Packet Lengths
            "mean_pkt_len_fwd": mean_pkt_len_fwd,
            "mean_pkt_len_bwd": mean_pkt_len_bwd,

            # Bytes/Packet Rates
            "fb_psec": fb_psec,
            "fp_psec": fp_psec,

            # Fwd/Bwd Ratios
            # "up_down_bytes_ratio": up_down_bytes_ratio,
            # "up_down_pkt_ratio": up_down_pkt_ratio,

            # FIAT
            "total_fiat": fiat_total,
            "mean_fiat": fiat_mean,
            "median_fiat": fiat_median,
            "min_fiat": fiat_min,
            "max_fiat": fiat_max,
            "std_fiat": fiat_std,

            # BIAT
            "total_biat": biat_total,
            "mean_biat": biat_mean,
            "median_biat": biat_median,
            "min_biat": biat_min,
            "max_biat": biat_max,
            "std_biat": biat_std,

            # Flow IAT
            "mean_flowiat": flowiat_mean,
            "median_flowiat": flowiat_median,
            "min_flowiat": flowiat_min,
            "max_flowiat": flowiat_max,
            "std_flowiat": flowiat_std,

            # Frequency-based
            "flowiat_skew": flowiat_skew,
            "flowiat_kurtosis": flowiat_kurt,
            "size_skewness": size_skewness,
            "size_kurtosis": size_kurtosis
        }

    return features