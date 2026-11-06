"""CPU-only evidence for continuous GPU idleness; no CUDA initialization."""
import math


def parse_snapshot(gpus, processes, policy):
    expected = {int(k): v for k, v in policy["allowed_gpus"].items()}
    busy = set()
    for line in processes.splitlines():
        if not line.strip():
            continue
        uuid, pid = [x.strip() for x in line.split(",")]
        if not uuid.startswith("GPU-") or int(pid) <= 0:
            raise ValueError("unverifiable GPU process")
        busy.add(uuid)
    result = {}
    for line in gpus.splitlines():
        index, uuid, used, free, utilization = [x.strip() for x in line.split(",")]
        index = int(index)
        if index not in expected:
            continue
        values = [float(used), float(free), float(utilization)]
        if (uuid != expected[index] or index in result
                or any(not math.isfinite(x) or x < 0 for x in values)
                or values[2] > 100):
            raise ValueError("unverifiable GPU identity or measurement")
        used, free, utilization = values
        result[index] = {"uuid": uuid, "used_mib": used, "free_mib": free,
                         "utilization_percent": utilization, "has_compute_process": uuid in busy,
                         "idle": uuid not in busy and used <= policy["max_used_mib"]
                         and free >= policy["minimum_free_mib"]
                         and utilization <= policy["max_utilization_percent"]}
    if set(result) != set(expected):
        raise ValueError("authorized GPU identity missing")
    return result


class IdleTracker:
    def __init__(self, policy):
        self.policy = policy
        self.reset()

    def reset(self):
        self.last = None
        self.since = {int(k): None for k in self.policy["allowed_gpus"]}
        self.counts = dict.fromkeys(self.since, 0)
        self.max_gaps = dict.fromkeys(self.since, 0)

    def observe(self, snapshot, now):
        if set(snapshot) != set(self.since):
            self.reset()
            raise ValueError("incomplete GPU observation")
        gap = 0 if self.last is None else now - self.last
        if gap < 0 or gap > self.policy["max_poll_gap_seconds"]:
            self.reset()
            gap = 0
        for index, row in snapshot.items():
            if row["idle"]:
                if self.since[index] is None:
                    self.since[index] = now
                    self.counts[index] = 0
                    self.max_gaps[index] = 0
                else:
                    self.max_gaps[index] = max(self.max_gaps[index], gap)
                self.counts[index] += 1
            else:
                self.since[index] = None
                self.counts[index] = 0
                self.max_gaps[index] = 0
        self.last = now
        return sorted((i for i, duration in self.durations(now).items()
                       if duration >= self.policy["idle_seconds"]),
                      key=lambda i: (self.since[i], i))

    def durations(self, now):
        return {i: 0 if since is None else now - since for i, since in self.since.items()}


def validate_ticket(ticket, policy, index, digest, boot_id, now):
    if (ticket["gpu_index"] != index
            or ticket["gpu_uuid"] != policy["allowed_gpus"][str(index)]
            or ticket["policy_sha256"] != digest or ticket["boot_id"] != boot_id
            or not 0 <= now - ticket["qualified_at_monotonic"] <= 45
            or not math.isfinite(ticket["idle_seconds"])
            or ticket["idle_seconds"] < policy["idle_seconds"]
            or ticket["sample_count"] < math.ceil(policy["idle_seconds"] / policy["max_poll_gap_seconds"]) + 1
            or not 0 <= ticket["max_gap_seconds"] <= policy["max_poll_gap_seconds"]):
        raise ValueError("invalid, expired or incomplete GPU idle ticket")
