#!/usr/bin/env python3
"""
Find cameras whose cloud UUID disagrees with the UUID the edge is using.

WHY THIS EXISTS
---------------
`InventoryService._adoption_entry()` used to drop `edge_camera_uuid` when a user
added a camera to a site from the inventory screen. The adopter then minted a
fresh UUID, so the cloud filed the camera under a UUID the Jetson had never
heard of. The result is a camera that looks fine — MediaMTX pulls the RTSP
source directly and never needs the two UUIDs to agree, so live video plays —
but which produces **no detections at all**, because the cloud subscribes to
SSE for its own UUID while the edge only ever emits the original one.

That bug is fixed for new adds. This script finds the cameras created *before*
the fix, which are not migrated automatically.

WHAT IT CHECKS
--------------
For every `camera_inventory` row that was added to a site (state='added') and
carries an `edge_camera_uuid`, compare that value with the `camera_uuid` of the
`camera` row it points at. A mismatch is a broken camera.

Note on types: `camera.camera_uuid` is a GUID column (BINARY(16) on MySQL) while
`camera_inventory.edge_camera_uuid` is free text. A raw SQL join between them
would never match on MySQL, which is why this reads through the ORM and compares
normalized `uuid.UUID` values in Python.

READ-ONLY BY DEFAULT. `--fix` is deliberately not implemented: re-pointing a
camera means re-creating its channel, MediaMTX path and edge registration, and
the right remedy depends on whether the edge still has the original UUID
running. See the remediation notes this script prints.

Examples:
  python3 Backend/scripts/audit_inventory_camera_uuids.py
  python3 Backend/scripts/audit_inventory_camera_uuids.py --all
  python3 Backend/scripts/audit_inventory_camera_uuids.py --json > mismatches.json
  DATABASE_URL='Driver={MySQL ODBC 8.0 Unicode Driver};Server=...' \
      python3 Backend/scripts/audit_inventory_camera_uuids.py
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import uuid as uuid_mod
from typing import Any, Dict, List, Optional

# Import the app's own DB configuration so this script talks to exactly the
# database the service uses, on MySQL or MSSQL, without duplicating the DSN
# parsing. Requires running from a checkout with Backend/ importable.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from sqlalchemy import select
    from core.database import db_manager
    from core.database_orm import Camera, CameraInventory, INVENTORY_ADDED
except ImportError as exc:  # pragma: no cover - environment problem, not logic
    print(
        "Could not import the backend package. Run this from the repository "
        "with Backend/ on the path and its requirements installed.\n"
        f"  {exc}",
        file=sys.stderr,
    )
    raise SystemExit(2) from exc


def _as_uuid(value: Any) -> Optional[uuid_mod.UUID]:
    """Normalize a UUID-ish value, or None when it is absent/unparseable.

    `edge_camera_uuid` is free text, so it can legitimately hold rubbish. A bad
    value is reported as its own category rather than crashing the audit.
    """
    if value is None:
        return None
    if isinstance(value, uuid_mod.UUID):
        return value
    text = str(value).strip()
    if not text:
        return None
    try:
        return uuid_mod.UUID(text)
    except ValueError:
        return None


async def collect(include_all: bool) -> Dict[str, List[Dict[str, Any]]]:
    """Classify every relevant inventory row.

    Buckets:
      mismatched   - cloud UUID != edge UUID. These are the broken cameras.
      unparseable  - edge_camera_uuid holds a value that is not a UUID.
      no_edge_uuid - inventory never recorded an edge UUID (legacy row).
      dangling     - inventory points at a camera row that no longer exists.
      ok           - the two agree (only collected with --all).
    """
    out: Dict[str, List[Dict[str, Any]]] = {
        "mismatched": [], "unparseable": [], "no_edge_uuid": [],
        "dangling": [], "ok": [],
    }

    if db_manager.AsyncSessionLocal is None:
        raise RuntimeError("Async database session factory is not configured")

    async with db_manager.AsyncSessionLocal() as db:
        rows = (
            await db.execute(
                select(CameraInventory).where(
                    CameraInventory.state == INVENTORY_ADDED,
                    CameraInventory.camera_uuid.isnot(None),
                )
            )
        ).scalars().all()

        for row in rows:
            cloud_uuid = _as_uuid(row.camera_uuid)
            edge_uuid = _as_uuid(row.edge_camera_uuid)

            cam = (
                await db.execute(
                    select(Camera).where(Camera.camera_uuid == row.camera_uuid)
                )
            ).scalar_one_or_none()

            record = {
                "discovery_identity": row.discovery_identity,
                "device_uuid": str(row.device_uuid) if row.device_uuid else None,
                "site_uuid": str(row.site_uuid) if row.site_uuid else None,
                "cloud_camera_uuid": str(cloud_uuid) if cloud_uuid else None,
                "edge_camera_uuid_raw": row.edge_camera_uuid,
                "camera_name": getattr(cam, "name", None),
                "camera_code": getattr(cam, "camera_code", None),
                "detection_enabled": bool(getattr(cam, "is_detection_enabled", False)) if cam else None,
            }

            if cam is None:
                out["dangling"].append(record)
            elif row.edge_camera_uuid and edge_uuid is None:
                out["unparseable"].append(record)
            elif edge_uuid is None:
                out["no_edge_uuid"].append(record)
            elif edge_uuid != cloud_uuid:
                record["edge_camera_uuid"] = str(edge_uuid)
                out["mismatched"].append(record)
            elif include_all:
                out["ok"].append(record)

    return out


def _print_group(title: str, rows: List[Dict[str, Any]], note: str) -> None:
    print(f"\n{title}: {len(rows)}")
    if not rows:
        return
    print(f"  {note}")
    for r in rows:
        label = r.get("camera_name") or r.get("camera_code") or "(unnamed)"
        print(f"  - {label}  [{r['discovery_identity']}]")
        print(f"      cloud camera_uuid : {r['cloud_camera_uuid']}")
        print(f"      edge  camera_uuid : {r.get('edge_camera_uuid') or r['edge_camera_uuid_raw']!r}")
        if r.get("detection_enabled") is False:
            print("      note: detection is DISABLED on this camera anyway")


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Audit cloud cameras against the edge UUIDs recorded in inventory.",
    )
    ap.add_argument("--all", action="store_true",
                    help="also list cameras whose UUIDs agree")
    ap.add_argument("--json", action="store_true",
                    help="emit machine-readable JSON instead of a report")
    args = ap.parse_args()

    async def run() -> Dict[str, List[Dict[str, Any]]]:
        try:
            return await collect(include_all=args.all)
        finally:
            # Without this the pooled connections keep the loop's resources
            # alive and the process hangs after printing its report instead of
            # exiting. A short-lived script must close what it opened.
            engine = getattr(db_manager, "async_engine", None)
            if engine is not None:
                await engine.dispose()

    try:
        result = asyncio.run(run())
    except Exception as exc:
        print(f"Audit failed: {exc}", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(result, indent=2, default=str))
        return 1 if result["mismatched"] else 0

    total = sum(len(v) for v in result.values())
    print("=" * 72)
    print("Inventory / cloud camera UUID audit")
    print("=" * 72)
    print(f"Inventory rows marked 'added' with a linked camera: {total}")

    _print_group(
        "MISMATCHED (broken: video plays, no detections ever arrive)",
        result["mismatched"],
        "The cloud subscribes to its own UUID; the edge emits the other one.",
    )
    _print_group(
        "UNPARSEABLE edge_camera_uuid",
        result["unparseable"],
        "Inventory holds a non-UUID value. Re-run device inventory refresh.",
    )
    _print_group(
        "DANGLING inventory rows",
        result["dangling"],
        "Inventory points at a camera row that no longer exists.",
    )

    print(f"\nNo edge UUID recorded (legacy rows, not necessarily broken): "
          f"{len(result['no_edge_uuid'])}")
    if args.all:
        print(f"Healthy (UUIDs agree): {len(result['ok'])}")

    if result["mismatched"]:
        print("\n" + "-" * 72)
        print("REMEDIATION")
        print("-" * 72)
        print(
            "For each mismatched camera, confirm on the Jetson which UUID is\n"
            "actually running before changing anything:\n"
            "\n"
            "    curl -s http://<device_url>/cameras | python3 -m json.tool\n"
            "\n"
            "The edge UUID above should appear there. Then pick one:\n"
            "\n"
            "  A. Delete the cloud camera and re-add it from the inventory\n"
            "     screen. With the fix in place the edge UUID is now reused,\n"
            "     so the camera comes back correctly wired. Simplest, but it\n"
            "     loses that camera's ROI, name and notification settings.\n"
            "\n"
            "  B. Re-point the cloud row in place, keeping its settings. This\n"
            "     means updating camera.camera_uuid and every row referencing\n"
            "     it (channel_configurations, pipeline_cameras, wall_cameras,\n"
            "     notification, video_record, camera_inventory). Do it in one\n"
            "     transaction, take a backup first, and expect camera_code to\n"
            "     change, which also changes the MediaMTX path.\n"
            "\n"
            "Option A is recommended unless the camera carries ROI or alert\n"
            "configuration worth preserving.\n"
        )
        return 1

    print("\nNo UUID mismatches found.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
