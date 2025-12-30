from typing import Dict, Iterator, Tuple, List, Any
from dataclasses import dataclass, field


# =========================
# FlowKey (bidirectional)
# =========================
@dataclass(frozen=True)
class FlowKey:
    ip1: str
    port1: int
    ip2: str
    port2: int
    protocol: str

    @staticmethod
    def from_packet(pkt: dict) -> "FlowKey":
        a = (pkt["ip_src"], pkt.get("sport", 0))
        b = (pkt["ip_dst"], pkt.get("dport", 0))
        if a <= b:
            return FlowKey(a[0], a[1], b[0], b[1], pkt["protocol"])
        else:
            return FlowKey(b[0], b[1], a[0], a[1], pkt["protocol"])


# =========================
# BasicFlow (ca în Java)
# =========================
@dataclass
class BasicFlow:
    key: FlowKey
    start_time: float
    last_seen: float
    packets: List[dict] = field(default_factory=list)

    def add_packet(self, pkt: dict):
        self.packets.append(pkt)
        self.last_seen = pkt["timestamp"]

    def packet_count(self) -> int:
        return len(self.packets)


# =========================
# Flow Generator
# =========================
def aggregate_flows(
    normalized_packets,
    flow_timeout: float = 600.0,
    include_packets: bool = True
):
    """
    ISCX / CICFlowMeter-like flow aggregation
    (adapted to keep the original return format)
    """

    # flow_key -> list of packets
    active_flows = {}

    # flow_key -> flow start timestamp
    flow_start_time = {}

    # flow_key -> init direction
    flow_init_direction = {}

    for flow_key, packet in normalized_packets:
        ts = packet["timestamp"]

        # =========================
        # Flow exists
        # =========================
        if flow_key in active_flows:
            start_ts = flow_start_time[flow_key]

            # ---- Flow timeout (lifetime, ca la autori) ----
            if ts - start_ts > flow_timeout:
                pkts = active_flows.pop(flow_key)
                init_dir = flow_init_direction.pop(flow_key)
                flow_start_time.pop(flow_key)

                if len(pkts) > 1:
                    start_ts_f = pkts[0]["timestamp"]
                    end_ts_f = pkts[-1]["timestamp"]

                    yield {
                        "flow_key": flow_key,
                        "start_time": start_ts_f,
                        "end_time": end_ts_f,
                        "duration": end_ts_f - start_ts_f,
                        "packet_count": len(pkts),
                        "total_bytes": sum(
                            p.get("length", p.get("ip_len", p.get("pkt_len", 0)))
                            for p in pkts
                        ),
                        "init_dir": init_dir,
                        "protocol": flow_key.protocol if hasattr(flow_key, "protocol") else flow_key[4],
                        **({"packets": pkts} if include_packets else {})
                    }

                # start flow nou
                active_flows[flow_key] = [packet]
                flow_start_time[flow_key] = ts
                flow_init_direction[flow_key] = (
                    packet.get("ip_src"),
                    packet.get("sport", packet.get("port"))
                )
                continue

            # ---- FIN / RST închide flow-ul ----
            if packet.get("protocol") == "TCP":
                flags = packet.get("tcp_flags", {})
                if flags.get("FIN") or flags.get("RST"):
                    active_flows[flow_key].append(packet)

                    pkts = active_flows.pop(flow_key)
                    init_dir = flow_init_direction.pop(flow_key)
                    flow_start_time.pop(flow_key)

                    if len(pkts) > 1:
                        start_ts_f = pkts[0]["timestamp"]
                        end_ts_f = pkts[-1]["timestamp"]

                        yield {
                            "flow_key": flow_key,
                            "start_time": start_ts_f,
                            "end_time": end_ts_f,
                            "duration": end_ts_f - start_ts_f,
                            "packet_count": len(pkts),
                            "total_bytes": sum(
                                p.get("length", p.get("ip_len", p.get("pkt_len", 0)))
                                for p in pkts
                            ),
                            "init_dir": init_dir,
                            "protocol": flow_key.protocol if hasattr(flow_key, "protocol") else flow_key[4],
                            **({"packets": pkts} if include_packets else {})
                        }
                    continue

            # ---- Normal packet ----
            active_flows[flow_key].append(packet)

        # =========================
        # New flow
        # =========================
        else:
            active_flows[flow_key] = [packet]
            flow_start_time[flow_key] = ts
            flow_init_direction[flow_key] = (
                packet.get("ip_src"),
                packet.get("sport", packet.get("port"))
            )

    # =========================
    # Flush remaining flows
    # =========================
    for flow_key in list(active_flows.keys()):
        pkts = active_flows.pop(flow_key)
        init_dir = flow_init_direction.pop(flow_key)
        flow_start_time.pop(flow_key)

        if len(pkts) > 1:
            start_ts_f = pkts[0]["timestamp"]
            end_ts_f = pkts[-1]["timestamp"]

            yield {
                "flow_key": flow_key,
                "start_time": start_ts_f,
                "end_time": end_ts_f,
                "duration": end_ts_f - start_ts_f,
                "packet_count": len(pkts),
                "total_bytes": sum(
                    p.get("length", p.get("ip_len", p.get("pkt_len", 0)))
                    for p in pkts
                ),
                "init_dir": init_dir,
                "protocol": flow_key.protocol if hasattr(flow_key, "protocol") else flow_key[4],
                **({"packets": pkts} if include_packets else {})
            }