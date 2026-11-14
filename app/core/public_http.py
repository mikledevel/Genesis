"""HTTP к проверенному публичному IP без повторного DNS и неявных proxy."""
import http.client
import ipaddress
import socket
import ssl
from urllib.parse import urlsplit


class PublicHTTPError(ValueError):
    pass


def parse_public_url(url):
    try:
        parsed = urlsplit(url)
        if (parsed.scheme not in {"http", "https"} or not parsed.hostname
                or parsed.username is not None or parsed.password is not None
                or parsed.port not in (None, 80, 443) or any(ord(c) < 33 for c in url)):
            raise ValueError
        host = parsed.hostname.encode("idna").decode("ascii")
        try:
            if not ipaddress.ip_address(host).is_global:
                raise ValueError
        except ValueError:
            if host.replace(".", "").isdigit() or ":" in host or host.lower() == "localhost":
                raise PublicHTTPError("URL must address a public host")
        return parsed, host
    except (ValueError, TypeError, UnicodeError) as error:
        raise PublicHTTPError("A public HTTP(S) URL on port 80 or 443 is required") from error


def request_public(url, method="GET", headers=None, body=None, timeout=10, max_bytes=2_000_000, ips=None):
    parsed, host = parse_public_url(url)
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    addresses = ips if ips is not None else list({row[4][0] for row in socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)})
    if not addresses or any(not ipaddress.ip_address(ip).is_global for ip in addresses):
        raise PublicHTTPError("Host resolves to a disallowed address")
    connection = http.client.HTTPConnection(host, port, timeout=timeout)
    # Соединение идёт к уже проверенному IP; Host и TLS проверяются для исходного имени.
    sock = socket.create_connection((addresses[0], port), timeout=timeout)
    try:
        if parsed.scheme == "https":
            sock = ssl.create_default_context().wrap_socket(sock, server_hostname=host)
        connection.sock = sock
        safe_headers = {k: v for k, v in (headers or {}).items()
                        if k.lower() not in {"host", "connection", "transfer-encoding", "content-length"}}
        target = parsed.path or "/"
        if parsed.query:
            target += "?" + parsed.query
        connection.request(method.upper(), target, body=body, headers=safe_headers)
        response = connection.getresponse()
        if 300 <= response.status < 400:
            raise PublicHTTPError("Redirects are not permitted; use the final URL")
        raw = response.read(max_bytes + 1)
        if len(raw) > max_bytes:
            raise PublicHTTPError("Response exceeded the byte limit")
        return {"status": response.status, "body": raw, "headers": dict(response.getheaders())}
    finally:
        connection.close()
        sock.close()
