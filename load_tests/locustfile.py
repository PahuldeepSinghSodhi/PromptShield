"""Load test for the gateway.

    locust -f load_tests/locustfile.py --headless -u 50 -r 10 -t 60s --host http://localhost:8000 --csv reports/runtime/locust
"""

import random

from locust import HttpUser, between, task

from promptshield.data.synth import generate_pii_records

PROMPTS = generate_pii_records(500, seed=123)["text"].tolist()
HEADERS = {"X-API-Key": "dev-key"}


class GatewayUser(HttpUser):
    wait_time = between(0.05, 0.3)

    @task(5)
    def scan(self):
        self.client.post("/v1/scan", json={"text": random.choice(PROMPTS)}, headers=HEADERS, name="/v1/scan")

    @task(3)
    def proxy(self):
        self.client.post(
            "/v1/proxy/chat",
            json={"messages": [{"role": "user", "content": random.choice(PROMPTS)}]},
            headers=HEADERS, name="/v1/proxy/chat",
        )

    @task(1)
    def long_document(self):
        doc = " ".join(random.sample(PROMPTS, 40))  # ~3k chars -> multiple windows
        self.client.post("/v1/redact", json={"text": doc}, headers=HEADERS, name="/v1/redact (long)")
