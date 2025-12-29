# src/flow_aggregator.py

from typing import Iterator, Dict, Any, List, Tuple
from pcap_reader import FlowKey, ParsedPacket, NormalizedPacket # Importăm tipurile

# Tipul de date pentru a reprezenta un flux complet
CompleteFlow = Dict[str, Any]

def aggregate_flows(normalized_packets: Iterator[NormalizedPacket], 
                    timeout: float = 15.0,
                    include_packets: bool = True) -> Iterator[CompleteFlow]:
    """
    Generator care reconstruiește fluxurile complete dintr-un stream de pachete,
    aplicând logica de timeout de 600 de secunde, specifică ISCX. [4, 5]

    By default this yields flow-level summary fields (scalars) suitable for CSV export.
    Set include_packets=True to keep the full 'packets' list in the output (not recommended for CSV).
    """
    # 1. Starea fluxurilor active
    active_flows: Dict[FlowKey, List[ParsedPacket]] = {}
    
    # 2. Direcția de inițiere a fluxului
    flow_init_direction: Dict[FlowKey, Tuple[str, int]] = {}
    
    for flow_key, packet in normalized_packets:
        current_time = packet['timestamp']

        # Verifică și termină fluxurile inactive (Bazat pe Timeout)
        keys_to_close = []
        for key, pkts in active_flows.items():
            if pkts and (current_time - pkts[-1]['timestamp']) > timeout:
                keys_to_close.append(key)

        for key in keys_to_close:
            pkts = active_flows.pop(key)
            init_dir = flow_init_direction.pop(key)

            # Compute summary fields to avoid writing whole lists into CSV cells
            start_ts = pkts[0]['timestamp']
            end_ts = pkts[-1]['timestamp']
            duration = end_ts - start_ts
            packet_count = len(pkts)
            # try common length fields, fallback to 0
            total_bytes = sum(p.get('length', p.get('ip_len', p.get('pkt_len', 0))) for p in pkts)
            if total_bytes == 0:
                for key, val in packet.items():
                    print(f"Packet field: {key} = {val}")

            flow_summary: CompleteFlow = {
                'flow_key': key,
                'start_time': start_ts,
                'end_time': end_ts,
                'duration': duration,
                'packet_count': packet_count,
                'total_bytes': total_bytes,
                'init_dir': init_dir,
                'protocol': key.protocol if hasattr(key, 'protocol') else key[4]
            }

            if include_packets:
                flow_summary['packets'] = pkts

            yield flow_summary

        # Adaugă pachetul curent
        if flow_key not in active_flows:
            # Inițiază un flux nou
            active_flows[flow_key] = []
            # Primul pachet definește direcția "Înainte" (Forward)
            flow_init_direction[flow_key] = (packet['ip_src'], packet.get('sport', packet.get('port')))

        active_flows[flow_key].append(packet)

    # La finalul PCAP-ului, închide toate fluxurile rămase (Flush)
    for flow_key in list(active_flows.keys()):
        pkts = active_flows.pop(flow_key)
        init_dir = flow_init_direction.pop(flow_key)

        start_ts = pkts[0]['timestamp']
        end_ts = pkts[-1]['timestamp']
        duration = end_ts - start_ts
        packet_count = len(pkts)
        total_bytes = sum(p.get('length', p.get('ip_len', p.get('pkt_len', 0))) for p in pkts)

        flow_summary: CompleteFlow = {
            'flow_key': flow_key,
            'start_time': start_ts,
            'end_time': end_ts,
            'duration': duration,
            'packet_count': packet_count,
            'total_bytes': total_bytes,
            'init_dir': init_dir,
            'protocol': key.protocol if hasattr(key, 'protocol') else key[4]
        }

        if include_packets:
            flow_summary['packets'] = pkts

        yield flow_summary