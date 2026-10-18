"""Verify that every local observability service is reachable and wired up.

The script probes each HTTP endpoint, pushes a probe gauge through the
Pushgateway, and reads the metric back from Prometheus to prove the whole
telemetry path works rather than only that the ports are open.
"""

from __future__ import annotations

import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass

from prometheus_client import CollectorRegistry, Gauge, push_to_gateway

PUSHGATEWAY_URL = "http://localhost:9091"
PROMETHEUS_URL = "http://localhost:9092"
PROBE_JOB = "eos_observability_probe"


@dataclass(frozen=True, slots=True)
class Endpoint:
    """A service health endpoint to probe.

    Args:
        name (str): Human-readable service name.
        url (str): Health URL returning any 2xx status when healthy.
    """

    name: str
    url: str


ENDPOINTS: tuple[Endpoint, ...] = (
    Endpoint("mlflow", "http://localhost:5001/health"),
    Endpoint("pushgateway", "http://localhost:9091/-/healthy"),
    Endpoint("prometheus", "http://localhost:9092/-/healthy"),
    Endpoint("grafana", "http://localhost:3001/api/health"),
)


def probe(endpoint: Endpoint, timeout: float = 5.0) -> bool:
    """Check a single health endpoint.

    Args:
        endpoint (Endpoint): Service to probe.
        timeout (float): Socket timeout in seconds.

    Returns:
        bool: True when the endpoint answers with a 2xx status.
    """
    try:
        with urllib.request.urlopen(endpoint.url, timeout=timeout) as response:
            status: int = response.status
            return 200 <= status < 300
    except (urllib.error.URLError, TimeoutError, ConnectionError):
        return False


def push_probe() -> None:
    """Push a probe gauge to the Pushgateway.

    Raises:
        Exception: Propagated when the Pushgateway is unreachable.
    """
    registry = CollectorRegistry()
    gauge = Gauge(
        "eos_observability_probe",
        "Observability pipeline health probe",
        registry=registry,
    )
    gauge.set(1.0)
    push_to_gateway(PUSHGATEWAY_URL, job=PROBE_JOB, registry=registry)


def query_probe(attempts: int = 8, delay: float = 5.0) -> float:
    """Read the probe value back from the Prometheus query API.

    Prometheus only ingests pushed samples on its next scrape, so the query is
    retried until the sample appears or the attempt budget is exhausted.

    Args:
        attempts (int): Maximum number of query attempts.
        delay (float): Seconds to sleep between attempts.

    Returns:
        float: Current probe value.

    Raises:
        RuntimeError: If Prometheus never returns a sample.
    """
    query = urllib.parse.quote(f'eos_observability_probe{{job="{PROBE_JOB}"}}')
    url = f"{PROMETHEUS_URL}/api/v1/query?query={query}"
    for attempt in range(1, attempts + 1):
        with urllib.request.urlopen(url, timeout=10) as response:
            payload = json.load(response)
        results = payload.get("data", {}).get("result", [])
        if results:
            return float(results[0]["value"][1])
        if attempt < attempts:
            time.sleep(delay)
    raise RuntimeError("prometheus has no sample for the observability probe")


def main() -> int:
    """Run every service probe and the end-to-end telemetry check.

    Returns:
        int: Process exit code, zero when every check succeeds.
    """
    failed = [endpoint for endpoint in ENDPOINTS if not probe(endpoint)]
    for endpoint in ENDPOINTS:
        state = "FAIL" if endpoint in failed else "OK"
        print(f"[{state}] {endpoint.name:<11} {endpoint.url}")
    if failed:
        return 1
    push_probe()
    value = query_probe()
    print(f"[OK] prometheus  probe value={value:.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
