"""Validate a LAN IP Webcam address for direct or paired local capture."""

from ipaddress import ip_address
from urllib.parse import urlparse


def ip_webcam_direct_source(raw_address: str) -> str:
    address = raw_address.strip()
    if not address:
        raise ValueError("Enter the current HTTP IPv4 address shown by IP Webcam on the Samsung.")
    if "://" not in address:
        address = "http://" + address
    parsed = urlparse(address)
    if parsed.scheme != "http":
        raise ValueError("Use the phone's HTTP address, for example http://10.73.3.83:8080. Do not use its HTTPS address here.")
    try:
        host = ip_address(parsed.hostname or "")
        port = parsed.port or 8080
    except ValueError as exc:
        raise ValueError("Enter the phone's numeric IPv4 address and port, for example http://10.73.3.83:8080.") from exc
    if (host.version != 4 or not host.is_private or host.is_loopback or host.is_unspecified
            or parsed.username or parsed.password or parsed.query or parsed.fragment
            or parsed.path not in ("", "/", "/video")):
        raise ValueError("Enter the phone's private HTTP IPv4 address and port, for example http://10.73.3.83:8080.")
    return f"http://{host}:{port}/video"


def ip_webcam_relay_source(raw_address: str) -> str:
    return "relay:" + ip_webcam_direct_source(raw_address)


def is_local_view(url: str | None) -> bool:
    """The same dashboard also has a hosted URL, where server cameras are remote."""
    return urlparse(url or "").hostname in {"127.0.0.1", "localhost", "::1"}
