# src/pcap_ingest.py (CORECTAT PENTRU INDEX ERROR)

from scapy.all import rdpcap, IP, TCP, UDP
from typing import Iterator, Dict, Any, Tuple

# Definirea tipurilor de date
FlowKey = Tuple[str,...]
ParsedPacket = Dict[str, Any]
NormalizedPacket = Tuple[FlowKey, ParsedPacket] 

def pcap_reader_generator(filepath: str) -> Iterator[ParsedPacket]:
    """
    Generator care citește pachetele dintr-un fișier PCAP și extrage metadatele de bază.
    """
    try:
        packets = rdpcap(filepath) 
    except FileNotFoundError:
        print(f"Eroare: Fișierul PCAP nu a fost găsit la {filepath}")
        return
    except Exception as e:
        print(f"Eroare la citirea PCAP: {e}")
        return

    for pkt in packets:
        if IP not in pkt:
            continue

        packet_info = {
            'timestamp': float(pkt.time),
            'ip_src': pkt[IP].src,
            'ip_dst': pkt[IP].dst,
            'proto': pkt[IP].proto,
            'pkt_len': len(pkt),
        }

        if pkt.haslayer(TCP):
            packet_info['sport'] = pkt.sport
            packet_info['dport'] = pkt.dport
        elif pkt.haslayer(UDP):
            packet_info['sport'] = pkt.sport
            packet_info['dport'] = pkt.dport
        else:
            continue

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