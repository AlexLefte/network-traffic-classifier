# src/pcap_ingest.py (CORECTAT PENTRU INDEX ERROR)

from scapy.all import PcapReader, rdpcap, IP, TCP, UDP
from typing import Iterator, Dict, Any, Tuple

# Definirea tipurilor de date
FlowKey = Tuple[str,...]
ParsedPacket = Dict[str, Any]
NormalizedPacket = Tuple[FlowKey, ParsedPacket] 

def pcap_reader_generator(filepath: str):
    with PcapReader(filepath) as pcap:
        for pkt in pcap:
            if IP not in pkt:
                continue
            if not (pkt.haslayer(TCP) or pkt.haslayer(UDP)):
                continue  # skip non-TCP/UDP

            packet_info = {
                'timestamp': float(pkt.time),
                'ip_src': pkt[IP].src,
                'ip_dst': pkt[IP].dst,
                'proto': 6 if pkt.haslayer(TCP) else 17,
                'pkt_len': len(pkt),
                'sport': pkt.sport,
                'dport': pkt.dport
            }

            yield packet_info

def normalize_flow_key(packet: ParsedPacket) -> FlowKey:
    """
    Generează o cheie canonică bidirecțională de flux (5-tuple).
    Cheia este sortată lexicografic (IP + Port). [3]
    """
    # Tuplă (IP, Port)
    src_tuple = (packet['ip_src'], packet['sport'])
    dst_tuple = (packet['ip_dst'], packet['dport'])
    
    # Sortare canonică pe baza IP și Port
    if src_tuple > dst_tuple:
        # Ordine: IP_min, Port_min, IP_max, Port_max, Protocol
        canonical_elements = (
            str(dst_tuple), str(dst_tuple[1]), 
            str(src_tuple), str(src_tuple[1]), 
            str(packet['proto'])
        )
    else:
        # Ordine: IP_min, Port_min, IP_max, Port_max, Protocol
        canonical_elements = (
            str(src_tuple), str(src_tuple[1]), 
            str(dst_tuple), str(dst_tuple[1]), 
            str(packet['proto'])
        )

    # Returnează o tuplă de string-uri (FlowKey)
    return tuple(canonical_elements)