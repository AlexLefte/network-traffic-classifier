import argparse
import os
import pandas as pd

from flow_aggregator import aggregate_flows
from feature_extractor import extract_features_from_flow
from pcap_reader import pcap_reader_generator, normalize_flow_key
from tqdm import tqdm

# Text to Index and Index to Text mappings
T2I = {
    # CHAT
    "aim": 0,
    "facebook_chat": 1,
    "gmail": 21,
    "hangouts_chat": 2,
    "skype_chat": 3,
    "icq": 4,

    # EMAIL
    "email": 5,

    # FILE TRANSFER
    "ftps": 6,
    "scp": 7,
    "sftp": 8,
    "skype_file": 9,

    # STREAMING
    "netflix": 10,
    "spotify": 11, 
    "vimeo" : 12,
    "youtube": 13,

    # VOIP
    "facebook_audio": 14,
    "hangouts_audio": 15,
    "skype_audio": 16,
    "voipbuster": 17,

    # VIDEO CALL
    "facebook_video": 18,
    "skype_video": 19,
    "hangouts_video": 20
}
I2T = {v: k for k, v in T2I.items()}


# =========================
# Main Processing Function
# =========================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pcap", required=True)
    parser.add_argument("--flow_interval", default=None)
    parser.add_argument("--output_csv", required=True)
    args = parser.parse_args()

    # Read PCAP files
    if os.path.isdir(args.pcap):
        pcap_files = [
            os.path.join(args.pcap, f)
            for f in os.listdir(args.pcap)
            if f.endswith(".pcap")
        ]
    else:
        pcap_files = [args.pcap]
    print(f"Found {len(pcap_files)} PCAP files to process.")

    # Flows counter
    total_flows = 0

    # Sort files ascending by size
    pcap_files.sort(key=lambda f: os.path.getsize(f))
    # pcap_files = pcap_files[-4:]

    # Process each PCAP file
    for pcap_path in tqdm(pcap_files):
        # Get traffic type from filename
        pcap_name = os.path.basename(pcap_path).lower()
        traffic_type = None
        for k in T2I.keys():
            if k in pcap_name:
                traffic_type = T2I[k]
                break
        if traffic_type is None:
            raise Exception(f"Unknown traffic type in filename: {pcap_name}. Supported types: {list(T2I.keys())}.")
        print(f"Processing {pcap_path} as traffic type '{I2T[traffic_type]}'...")

        # Read packets and normalize flow keys
        packets_generator = (
            (normalize_flow_key(pachet_dict), pachet_dict)
            for pachet_dict in pcap_reader_generator(pcap_path)
        )
        
        # Aggregate flows
        if args.flow_interval is not None:
            flow_interval = float(args.flow_interval)
        else:
            flow_interval = float('inf')
        flows = aggregate_flows(packets_generator, 
                                flow_timeout=flow_interval)
        rows = []
        for flow in flows:
            # print(f"  Flow: {flow['flow_key']} | Duration: {flow['duration']} | Init Dir: {flow['init_dir']} | Packets: {flow['packet_count']} | Bytes: {flow['total_bytes']} | Protocol: {flow['protocol']}")
            feats = extract_features_from_flow(flow, file=pcap_name)
            feats["Label"] = traffic_type
            rows.append(feats)

        # Save features to CSV
        df = pd.DataFrame(rows)
        if os.path.exists(args.output_csv):
            df.to_csv(args.output_csv, mode="a", header=False, index=False)
        else:
            df.to_csv(args.output_csv, index=False)

        # Increment flows count
        total_flows += len(df)
        print(f"[OK] {len(df)} flows appended to {args.output_csv}")
    
    # Print toal flows count
    print(f"Total flows processed: {total_flows}") 

if __name__ == "__main__":
    main()