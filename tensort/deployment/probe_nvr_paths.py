#!/usr/bin/env python3
"""Find the RTSP path an NVR actually serves for each of its cameras.

Why this exists
---------------
ISAPI reports an NVR's proxied cameras as InputProxyChannel id 1..N, but the
NVR's RTSP server does not necessarily stream them on those same numbers —
PoE/proxied inputs are classically offset (id 1 served as channel 33). Asking
for the wrong path returns an RTSP 404, which GStreamer surfaces only as

    module rtspsrc0 reported: Not found
    unable to start pipeline

...which looks like a pipeline bug rather than a wrong URL. Rather than guess
the offset by editing .env and restarting the service, this talks RTSP directly
and reports which paths answer.

Usage
-----
    python3 deployment/probe_nvr_paths.py <host> <port> <user> <pass> [channels]

    # McCallum Office, channels 1-15:
    python3 deployment/probe_nvr_paths.py \\
        initialsecurity.dvrlists.com 16805 monitor1 'SECRET' 1-15

It sends a bare RTSP DESCRIBE per candidate path (no video is decoded, so it is
cheap and safe to run against a live NVR) and prints the response code. Set
NVR_CHANNEL_OFFSET in .env from whatever comes back 200 OK.
"""

import base64
import hashlib
import re
import socket
import sys
import uuid

TIMEOUT_S = 6.0
# Offsets worth trying: none, the classic PoE offset, and a couple seen in the
# wild on larger chassis.
OFFSETS = (0, 32, 16)
STREAM = 2  # substream; matches HIK_RTSP_STREAM


def _parse_channels(spec):
    out = []
    for part in str(spec).split(","):
        part = part.strip()
        if "-" in part:
            lo, _, hi = part.partition("-")
            if lo.isdigit() and hi.isdigit():
                out.extend(range(int(lo), int(hi) + 1))
        elif part.isdigit():
            out.append(int(part))
    return out or [1]


def _digest_header(method, uri, user, pwd, challenge):
    """Build an RFC 2617 digest Authorization header."""
    def field(name):
        m = re.search(r'%s="?([^",]+)"?' % name, challenge)
        return m.group(1) if m else ""

    realm, nonce = field("realm"), field("nonce")
    qop, opaque = field("qop"), field("opaque")

    def md5(s):
        return hashlib.md5(s.encode()).hexdigest()

    ha1 = md5("%s:%s:%s" % (user, realm, pwd))
    ha2 = md5("%s:%s" % (method, uri))
    cnonce, nc = uuid.uuid4().hex[:16], "00000001"

    if qop:
        resp = md5("%s:%s:%s:%s:%s:%s" % (ha1, nonce, nc, cnonce, qop, ha2))
        auth = ('Digest username="%s", realm="%s", nonce="%s", uri="%s", '
                'response="%s", qop=%s, nc=%s, cnonce="%s"'
                % (user, realm, nonce, uri, resp, qop, nc, cnonce))
    else:
        resp = md5("%s:%s:%s" % (ha1, nonce, ha2))
        auth = ('Digest username="%s", realm="%s", nonce="%s", uri="%s", '
                'response="%s"' % (user, realm, nonce, uri, resp))
    if opaque:
        auth += ', opaque="%s"' % opaque
    return auth


def describe(host, port, user, pwd, path):
    """Send RTSP DESCRIBE for one path. Returns (code, reason).

    The challenge and the authenticated retry MUST share one TCP connection.
    Hikvision's HIK Media Server binds the digest nonce to the connection, so
    reconnecting for the retry invalidates it and every path answers 401 — which
    reads as "wrong password" when the credentials are in fact correct.
    """
    uri = "rtsp://%s:%s%s" % (host, port, path)

    def request(sock, seq, extra_auth=None):
        req = "DESCRIBE %s RTSP/1.0\r\nCSeq: %d\r\n" % (uri, seq)
        req += "Accept: application/sdp\r\n"
        if extra_auth:
            req += "Authorization: %s\r\n" % extra_auth
        req += "\r\n"
        sock.sendall(req.encode())
        return sock.recv(4096).decode("utf-8", "ignore")

    try:
        sock = socket.create_connection((host, int(port)), TIMEOUT_S)
    except Exception as e:
        return None, "connection failed: %s" % e

    try:
        sock.settimeout(TIMEOUT_S)
        resp = request(sock, 1)

        first = resp.split("\r\n", 1)[0]
        if " 401 " not in first:
            return _code(first)

        # Answer the auth challenge: digest first, basic as a fallback.
        challenge = ""
        for line in resp.split("\r\n"):
            if line.lower().startswith("www-authenticate:"):
                challenge = line
                break

        if "digest" in challenge.lower():
            auth = _digest_header("DESCRIBE", uri, user, pwd, challenge)
        else:
            token = base64.b64encode(("%s:%s" % (user, pwd)).encode()).decode()
            auth = "Basic %s" % token

        # Same socket, next CSeq — this is the part that matters.
        return _code(request(sock, 2, auth).split("\r\n", 1)[0])
    except Exception as e:
        return None, "auth failed: %s" % e
    finally:
        sock.close()


def _code(status_line):
    parts = status_line.split(None, 2)
    if len(parts) >= 2 and parts[1].isdigit():
        return int(parts[1]), (parts[2] if len(parts) > 2 else "")
    return None, status_line


def main():
    if len(sys.argv) < 5:
        print(__doc__)
        return 2

    host, port, user, pwd = sys.argv[1:5]
    channels = _parse_channels(sys.argv[5] if len(sys.argv) > 5 else "1-4")

    print("Probing %s:%s as %s" % (host, port, user))
    print("Channels: %s\n" % ", ".join(str(c) for c in channels))

    working = {}
    for offset in OFFSETS:
        hits = []
        for ch in channels:
            path = "/Streaming/Channels/%d" % ((ch + offset) * 100 + STREAM)
            code, reason = describe(host, port, user, pwd, path)
            if code == 200:
                hits.append(ch)
            marker = "OK " if code == 200 else "   "
            print("  %soffset=%-3d ch %-3d %-28s -> %s %s"
                  % (marker, offset, ch, path, code if code else "-", reason[:30]))
        if hits:
            working[offset] = hits
        print("")

    print("=" * 60)
    if not working:
        print("No path answered 200 OK.")
        print("Check credentials, that the port is the RTSP port (not HTTP),")
        print("and that the NVR allows this account to stream.")
        return 1

    best = max(working, key=lambda o: len(working[o]))
    print("Working offset: %d  (%d/%d channels answered)"
          % (best, len(working[best]), len(channels)))
    print("\nSet this in .env:\n    NVR_CHANNEL_OFFSET=%d" % best)
    return 0


if __name__ == "__main__":
    sys.exit(main())
