"""
Phase 1 Validation: Test STTM Desktop connectivity.

Run: python scripts/test_sttm_connection.py

Prerequisites: STTM Desktop must be running.

Tests:
1. Port discovery (find which port STTM is listening on)
2. HTTP GET to verify server response
3. HTTP POST to /api/bani-control (test payload)
"""

import asyncio
import sys

try:
    import httpx
except ImportError:
    print("ERROR: httpx not installed. Run: pip install httpx")
    sys.exit(1)


STTM_PORTS = [1397, 1469, 1539, 1552, 1574, 1581, 1606, 1644, 1661, 1665, 1675, 1708]


async def discover_sttm_port() -> str | None:
    """Try each known STTM port to find the active one."""
    print("1. Discovering STTM Desktop port...")
    print(f"   Trying ports: {STTM_PORTS}")

    async with httpx.AsyncClient() as client:
        for port in STTM_PORTS:
            url = f"http://localhost:{port}"
            try:
                resp = await client.get(url, timeout=1.0)
                print(f"   Port {port}: {resp.status_code} - FOUND!")
                return url
            except httpx.ConnectError:
                print(f"   Port {port}: not listening")
            except httpx.TimeoutException:
                print(f"   Port {port}: timeout")
            except Exception as e:
                print(f"   Port {port}: {type(e).__name__}: {e}")

    return None


async def test_sttm_endpoints(base_url: str):
    """Test known STTM endpoints."""
    print(f"\n2. Testing STTM endpoints at {base_url}...")

    endpoints = [
        ("GET", "/"),
        ("GET", "/api"),
        ("GET", "/api/bani-control"),
    ]

    async with httpx.AsyncClient() as client:
        for method, path in endpoints:
            url = f"{base_url}{path}"
            try:
                if method == "GET":
                    resp = await client.get(url, timeout=2.0)
                else:
                    resp = await client.post(url, json={}, timeout=2.0)
                print(f"   {method} {path}: {resp.status_code}")
                if resp.status_code == 200:
                    content_type = resp.headers.get("content-type", "")
                    body_preview = resp.text[:200] if len(resp.text) > 0 else "(empty)"
                    print(f"     Content-Type: {content_type}")
                    print(f"     Body: {body_preview}")
            except Exception as e:
                print(f"   {method} {path}: {type(e).__name__}: {e}")


async def test_bani_control_post(base_url: str):
    """Test POST to /api/bani-control with sample payloads."""
    print(f"\n3. Testing POST /api/bani-control...")

    # Try various payload formats that STTM might expect
    payloads = [
        {"type": "search", "query": "vjk"},
        {"action": "search", "data": "vjk"},
        {"shabadId": 1, "line": 0},
        {"type": "shabad", "id": 1},
    ]

    async with httpx.AsyncClient() as client:
        for payload in payloads:
            try:
                resp = await client.post(
                    f"{base_url}/api/bani-control",
                    json=payload,
                    timeout=2.0,
                )
                print(f"\n   Payload: {payload}")
                print(f"   Response: {resp.status_code} - {resp.text[:200]}")
            except Exception as e:
                print(f"\n   Payload: {payload}")
                print(f"   Error: {type(e).__name__}: {e}")


async def main():
    print("STTM Automate - STTM Connection Test")
    print("=" * 60)
    print("NOTE: Make sure STTM Desktop is running!\n")

    base_url = await discover_sttm_port()

    if not base_url:
        print("\n   STTM Desktop not found on any known port.")
        print("   Make sure STTM Desktop is running and try again.")
        print("\n   If STTM is running but not found, it might use a different port.")
        print("   Check STTM settings for the server port configuration.")
        return

    print(f"\n   STTM found at: {base_url}")

    await test_sttm_endpoints(base_url)
    await test_bani_control_post(base_url)

    print("\n\nDone! Review the output to understand:")
    print("  - Which port STTM uses")
    print("  - What endpoints are available")
    print("  - What payload format /api/bani-control expects")


if __name__ == "__main__":
    asyncio.run(main())
