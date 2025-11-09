# services/github-integration/src/metrics.py
from prometheus_client import Counter, Histogram, Gauge
import time

# Counter
AUTONOMOUS_REQUESTS = Counter(
    'autonomous_requests_total',
    'Total autonomous edit requests',
    ['status', 'risk_level']
)

# Histogram
GENERATION_LATENCY = Histogram(
    'kimi_generation_duration_seconds',
    'Time spent generating code',
    buckets=[1, 5, 10, 30, 60, 120, 300]
)

# Gauge
ACTIVE_REQUESTS = Gauge(
    'autonomous_active_requests',
    'Currently active requests'
)

async def track_request_duration():
    """Decorator für Latency-Tracking"""
    start = time.time()
    try:
        yield
    finally:
        GENERATION_LATENCY.observe(time.time() - start)