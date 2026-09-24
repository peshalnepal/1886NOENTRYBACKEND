#!/usr/bin/env python3
"""Explain, step by step, why camera discovery is or is not finding cameras.

Why this exists
---------------
A discovery sweep that finds nothing looks identical to a sweep that ran fine
on an empty network: both log "0 Hikvision camera(s) confirmed". The cause is
almost always one of a small number of boring things — .env not loaded, a wrong
password, ISAPI on a different port — and none of them are visible from the
service log alone.

This runs the same code the service runs, but narrates every step and prints the
actual HTTP status behind each decision.

Usage
-----
    # Check configuration and probe everything in .env:
    python3 deployment/diagnose_discovery.py

    # Probe one specific host (bypasses .env entirely):
    python3 deployment/diagnose_discovery.py --host 192.168.1.64
    python3 deployment/diagnose_discovery.py --host nvr.example.com \
        --http-port 8000 --rtsp-port 16805 --user monitor1 --password 'SECRET'

Safe to run against live hardware: it only reads ISAPI, never starts a stream
and never writes to the database.
"""

import argparse
import os
import socket
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
TENSORT_DIR = os.path.dirname(HERE)
sys.path.insert(0, TENSORT_DIR)

ENV_PATH = os.path.join(TENSORT_DIR, ".env")


def _load_env():
    """Load .env the way main.py does, reporting honestly which way it went."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return False, "python-dotenv is NOT installed"
    if not os.path.exists(ENV_PATH):
        return False, "no .env at {}".format(ENV_PATH)
    load_dotenv(ENV_PATH)
    return True, "loaded {}".format(ENV_PATH)


def _mask(secret):
    if not secret:
        return "(EMPTY)"
    return "*" * len(secret) + " ({} chars)".format(len(secret))


def _rule(title):
    print("\n" + "=" * 68)
    print(title)
    print("=" * 68)


def _check_port(host, port, timeout=3.0):
    """Is anything listening? Distinguishes 'wrong port' from 'wrong password'."""
    try:
        sock = socket.create_connection((host, int(port)), timeout)
        sock.close()
        return True, "open"
    except socket.timeout:
        return False, "timed out (filtered, or host unreachable)"
    except Exception as e:
        return False, str(e)


def report_config(discovery):
    _rule("1. CONFIGURATION")
    ok, how = _load_env()
    print("  .env            : {}".format(how))
    if not ok:
        print("\n  >>> The service reads its settings from the environment.")
        print("  >>> Without python-dotenv, `python3 main.py` runs on DEFAULTS:")
        print("  >>>   blank password, no NVRs -> discovery finds NOTHING.")
        print("  >>> Fix:  pip install python-dotenv")

    cfg = discovery.DiscoveryConfig()
    print("\n  DISCOVERY_ENABLED  : {}".format(cfg.enabled))
    print("  HIK_USERNAME       : {}".format(cfg.username))
    print("  HIK_PASSWORD       : {}".format(_mask(cfg.password)))
    print("  NVR_USERNAME       : {}".format(cfg.nvr_username))
    print("  NVR_PASSWORD       : {}".format(_mask(cfg.nvr_password)))
    print("  HIK_HTTP_PORT      : {}".format(cfg.http_port))
    print("  HIK_RTSP_PORT      : {}".format(cfg.rtsp_port))
    print("  NVR_CHANNEL_OFFSET : {}".format(cfg.nvr_channel_offset))
    print("  DISCOVERY_SUBNETS  : {}".format(cfg.subnets or "(derive from own IP)"))
    print("  DISCOVERY_NVRS     : {}".format(cfg.discovery_nvrs or "(none)"))

    problems = []
    if not cfg.enabled:
        problems.append("DISCOVERY_ENABLED is false — the scheduler will not run.")
    if not cfg.password and not cfg.discovery_nvrs:
        problems.append("HIK_PASSWORD is empty — every ISAPI probe will fail auth.")
    if cfg.discovery_nvrs and not cfg.nvr_password:
        problems.append("NVR_PASSWORD is empty but DISCOVERY_NVRS is set.")
    for p in problems:
        print("\n  PROBLEM: {}".format(p))
    return cfg


def probe_one(discovery, cfg, host, http_port, rtsp_port, credentials, channels):
    """Probe a single host and explain each outcome."""
    _rule("PROBING {}".format(host))
    print("  ISAPI (HTTP) port : {}".format(http_port))
    print("  RTSP port         : {}".format(rtsp_port))
    print("  username          : {}".format(credentials[0]))
    print("  password          : {}".format(_mask(credentials[1])))

    reachable, detail = _check_port(host, http_port)
    print("\n  TCP {}:{} -> {}".format(host, http_port, detail))
    if not reachable:
        print("\n  DIAGNOSIS: nothing is listening on the ISAPI port.")
        print("    - For a REMOTE NVR this is usually the wrong port: the 4th")
        print("      field of DISCOVERY_NVRS must be its HTTP port (the one the")
        print("      web UI uses), NOT its RTSP port.")
        print("    - For a LAN camera, check it is powered and on this subnet.")
        return []

    cameras = discovery.probe_device(
        host, cfg,
        http_port=http_port, rtsp_port=rtsp_port,
        credentials=credentials, channels=channels,
    )

    if not cameras:
        print("\n  DIAGNOSIS: the port is open but no camera was confirmed.")
        print("    Look for a 'HTTP 401' WARNING just above -> wrong username")
        print("    or password for this device.")
        print("    No 401 and no <DeviceInfo> -> not a Hikvision device, or")
        print("    ISAPI is disabled on it.")
        return []

    print("\n  CONFIRMED {} camera(s):".format(len(cameras)))
    for cam in cameras:
        url = cfg.rtsp_url_for(cam, force_credentials=credentials)
        print("\n    channel {:<3} {}".format(cam.channel, cam.device_name or "(unnamed)"))
        print("      model    : {}".format(cam.model or "unknown"))
        print("      serial   : {}".format(cam.serial_number or "(none)"))
        print("      identity : {}".format(discovery.identity_of(cam)))
        print("      stream   : {}".format(url))

    if cameras and cameras[0].is_nvr:
        print("\n  NOTE: these are NVR-proxied channels. If a stream later fails")
        print("        with 'reported: Not found', the RTSP channel numbering is")
        print("        offset. Find the right value with:")
        print("          python3 deployment/probe_nvr_paths.py {} {} <user> <pass> 1-16"
              .format(host, rtsp_port))
    return cameras


def main():
    parser = argparse.ArgumentParser(
        description="Diagnose Hikvision camera discovery.")
    parser.add_argument("--host", help="probe only this host, ignoring .env targets")
    parser.add_argument("--http-port", type=int, help="ISAPI port (default: HIK_HTTP_PORT)")
    parser.add_argument("--rtsp-port", type=int, help="RTSP port (default: HIK_RTSP_PORT)")
    parser.add_argument("--user", help="override username")
    parser.add_argument("--password", help="override password")
    parser.add_argument("--channels", help="NVR channels, e.g. 1-8")
    parser.add_argument("--full-sweep", action="store_true",
                        help="run a complete sweep, exactly as the service does")
    args = parser.parse_args()

    # Configure logging first: the probe's own WARNINGs (401s especially) are
    # half the diagnosis, so they must be visible.
    import logging
    logging.basicConfig(level=logging.INFO, format="  [%(levelname)s] %(message)s")

    import discovery

    cfg = report_config(discovery)

    if args.host:
        creds = (args.user or cfg.username, args.password or cfg.password)
        channels = discovery._parse_channel_spec(args.channels) if args.channels else None
        found = probe_one(
            discovery, cfg, args.host,
            args.http_port or cfg.http_port,
            args.rtsp_port or cfg.rtsp_port,
            creds, channels,
        )
        return 0 if found else 1

    if args.full_sweep:
        _rule("FULL SWEEP (this is exactly what the service runs)")
        found = discovery.scan_network(cfg)
        print("\n  {} camera(s) confirmed".format(len(found)))
        for cam in found:
            print("    {:<40} {}".format(discovery.identity_of(cam), cam.ip))
        return 0 if found else 1

    # Default: probe each configured remote NVR, then summarise the LAN plan.
    total = []
    for nvr in discovery.parse_remote_nvrs(
            cfg.discovery_nvrs,
            default_rtsp_port=cfg.rtsp_port,
            default_http_port=cfg.http_port):
        total.extend(probe_one(
            discovery, cfg, nvr.host, nvr.http_port, nvr.rtsp_port,
            (cfg.nvr_username, cfg.nvr_password), nvr.channels,
        ))

    _rule("2. LOCAL NETWORK")
    own = discovery.primary_ipv4()
    print("  This host's IPv4 : {}".format(own or "(could not determine)"))
    subnets = cfg.subnets or discovery._default_subnet_from_host(own)
    print("  Would sweep      : {}".format(", ".join(subnets) if subnets else "(nothing)"))
    print("\n  Re-run with --full-sweep to actually scan the LAN,")
    print("  or --host <ip> to test one camera.")

    _rule("SUMMARY")
    print("  {} camera(s) confirmed from configured NVRs".format(len(total)))
    return 0 if total else 1


if __name__ == "__main__":
    sys.exit(main())
