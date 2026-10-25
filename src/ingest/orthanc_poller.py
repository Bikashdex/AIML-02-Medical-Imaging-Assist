"""Poll Orthanc for newly-received studies and forward them to the inference API."""

from __future__ import annotations

import os
import time

import httpx

ORTHANC_URL = os.getenv("ORTHANC_URL", "http://localhost:8042")
API_URL = os.getenv("API_URL", "http://localhost:8000")
POLL_SECONDS = float(os.getenv("POLL_INTERVAL_SECONDS", "5"))
AUTH = None
if os.getenv("ORTHANC_USERNAME"):
    AUTH = (os.environ["ORTHANC_USERNAME"], os.environ["ORTHANC_PASSWORD"])


def fetch_dcm(client: httpx.Client, instance_id: str) -> bytes:
    return client.get(f"{ORTHANC_URL}/instances/{instance_id}/file", auth=AUTH).content


def send_to_api(client: httpx.Client, instance_id: str, payload: bytes) -> dict | None:
    try:
        response = client.post(
            f"{API_URL}/predict",
            files={"file": (f"{instance_id}.dcm", payload, "application/dicom")},
            timeout=60,
        )
        response.raise_for_status()
        return response.json()
    except httpx.HTTPError as exc:
        print(f"[ingest] {instance_id} -> API error: {exc}", flush=True)
        return None


def main() -> None:
    print(f"[ingest] watching {ORTHANC_URL} every {POLL_SECONDS:.0f}s", flush=True)
    last = 0
    with httpx.Client() as client:
        while True:
            try:
                changes = client.get(f"{ORTHANC_URL}/changes", params={"last": last}, auth=AUTH).json()
            except httpx.HTTPError as exc:
                print(f"[ingest] Orthanc unreachable: {exc}", flush=True)
                time.sleep(POLL_SECONDS)
                continue

            last = changes.get("Last", last)
            for instance in changes.get("Instances", []):
                if instance.get("Removed"):
                    continue
                instance_id = instance["ID"]
                result = send_to_api(client, instance_id, fetch_dcm(client, instance_id))
                if result:
                    print(
                        f"[ingest] {instance_id} -> {result['prediction']} "
                        f"p={result['probability']:.3f} ({result['inference_ms']}ms)",
                        flush=True,
                    )
            time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    main()
