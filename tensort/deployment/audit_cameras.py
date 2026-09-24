#!/usr/bin/env python3
"""Read-only comparison of saved captures with declared static NVR streams.

Does not open cameras or import the runtime/database initialization modules.
Usernames, passwords and URL query/fragment contents are never printed.
"""

import argparse
import json
import os
from pathlib import Path
import sqlite3
import sys
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from discovery import DiscoveryConfig, _static_nvr_cameras


def endpoint(url):
    parsed = urlsplit(url or "")
    if parsed.scheme.lower() != "rtsp" or not parsed.hostname:
        raise ValueError("Expected an RTSP source")
    return (parsed.scheme.lower(), parsed.hostname.lower(), parsed.port or 554,
            parsed.path, parsed.query)


def credentials(url):
    parsed = urlsplit(url)
    return unquote(parsed.username or ""), unquote(parsed.password or "")


def safe_endpoint(key):
    scheme, host, port, path, _query = key
    host = "[{}]".format(host) if ":" in host else host
    return "{}://{}:{}{}".format(scheme, host, port, path)


def audit_database(path, expected_urls):
    expected = {endpoint(url): url for url in expected_urls}
    # mode=ro prevents a typo in the path from creating an empty database.
    connection = sqlite3.connect(Path(path).resolve().as_uri() + "?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    results, seen = [], set()
    try:
        columns = {row[1] for row in connection.execute("PRAGMA table_info(camera_configs)")}
        source_column = "source_url" if "source_url" in columns else "rtsp_url"
        rows = connection.execute(
            "SELECT camera_uuid, config_json, {} AS source_url FROM camera_configs".format(source_column)
        )
        for row in rows:
            item = {"camera_uuid": row["camera_uuid"]}
            try:
                cfg = json.loads(row["config_json"] or "{}")
                # Startup restores config_json, not the redundant source column.
                url = cfg.get("source_url")
                key = endpoint(url)
                seen.add(key)
                enabled = cfg.get("enabled", True)
                if isinstance(enabled, str):
                    enabled = enabled.strip().lower() in ("1", "true", "yes", "on")
                item.update(
                    enabled=bool(enabled), endpoint=safe_endpoint(key),
                    matches_static_endpoint=key in expected,
                    credentials_match=(credentials(url) == credentials(expected[key])
                                       if key in expected else None),
                    source_column_matches_config=row["source_url"] == url,
                    settings={name: cfg.get(name) for name in (
                        "sample_fps", "resize", "decode_backend", "gst_decoder",
                        "gst_latency_ms", "rtsp_transport")},
                )
            except (ValueError, TypeError, AttributeError):
                item["error"] = "Missing/invalid saved config or RTSP source; contents withheld"
            results.append(item)
    finally:
        connection.close()
    return {
        "saved_count": len(results), "static_candidate_count": len(expected),
        "cameras": results,
        "static_endpoints_without_saved_config": [safe_endpoint(key) for key in expected if key not in seen],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", type=Path, default=ROOT / ".env")
    parser.add_argument("--db", type=Path, default=ROOT / "jetson_cameras.db")
    args = parser.parse_args()
    try:
        from dotenv import load_dotenv
        load_dotenv(str(args.env))  # Same precedence as main.py: exported env wins.
        cfg = DiscoveryConfig()
        expected_urls = [cfg.rtsp_url_for(dev) for dev in _static_nvr_cameras(cfg)]
        report = audit_database(args.db, expected_urls)
        report["configured_http_port"] = int(os.getenv("PORT", "8080"))
        report["max_cameras"] = int(os.getenv("MAX_CAMERAS", "8"))
    except Exception as exc:
        # Database/config errors can include raw inputs. Report only their type.
        print("Audit failed ({}); check env, database path and dependencies".format(type(exc).__name__))
        return 1
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
