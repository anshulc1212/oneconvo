import asyncio
import time
import requests
from recruitment import hr_flow


def deliver_webhook(webhook_url, payload, max_attempts=3):
    """POST payload to webhook_url with up to 3 retries (exponential backoff)."""
    for attempt in range(1, max_attempts + 1):
        try:
            resp = requests.post(webhook_url, json=payload, timeout=30)
            resp.raise_for_status()
            print(f"[Webhook] Delivered on attempt {attempt} → {resp.status_code}")
            return
        except Exception as e:
            print(f"[Webhook] Attempt {attempt} failed: {e}")
            if attempt < max_attempts:
                time.sleep(2 ** attempt)  # 2s, 4s backoff
    print(f"[Webhook] All {max_attempts} attempts exhausted for candidate {payload.get('candidate_id')}")


def process_and_notify(candidate_id, audio_url, webhook_url, client: str = "unknown"):  # ← NEW: client param
    """Run hr_flow in background thread, then POST result to webhook."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        result = loop.run_until_complete(hr_flow(audio_url, client))  # ← pass client
        payload = {
            "candidate_id": candidate_id,
            "status": "success",
            "source_url": audio_url,
            "result": result
        }
    except Exception as e:
        payload = {
            "candidate_id": candidate_id,
            "status": "error",
            "source_url": audio_url,
            "error": str(e)
        }
    finally:
        loop.close()

    deliver_webhook(webhook_url, payload)