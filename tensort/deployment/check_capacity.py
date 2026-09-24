"""Measure live camera throughput through /health; run after camera warm-up."""

import argparse
import json
import time
from urllib.request import urlopen


def read_health(url):
    with urlopen(url.rstrip("/") + "/health", timeout=5) as response:
        payload = json.load(response)
    if not payload.get("pipeline_ready"):
        raise RuntimeError("Pipeline is not ready")
    return payload["stats"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8080")
    parser.add_argument("--cameras", type=int, default=None,
                        help="Expected running cameras; defaults to the current capture count")
    parser.add_argument("--fps", type=float, default=3)
    parser.add_argument("--seconds", type=float, default=120)
    parser.add_argument("--max-age-ms", type=int, default=1000)
    args = parser.parse_args()
    if (args.cameras is not None and args.cameras < 1) or args.fps <= 0 or args.seconds <= 0:
        parser.error("Use a positive camera count, FPS and duration")

    baseline = read_health(args.url)
    camera_ids = set(baseline.get("capture", {}))
    if args.cameras is None:
        args.cameras = len(camera_ids)
    if args.cameras < 1 or args.cameras > baseline.get("max_cameras", args.cameras):
        parser.error("Expected cameras must be between one and the device's configured MAX_CAMERAS")
    if len(camera_ids) != args.cameras or not camera_ids <= baseline.get("cameras", {}).keys():
        raise SystemExit("Expected {} warmed-up cameras; check /health first".format(args.cameras))
    started = time.monotonic()
    max_ages = {key: 0 for key in camera_ids}
    issues = set()
    while time.monotonic() - started < args.seconds:
        time.sleep(min(1, max(0, args.seconds - (time.monotonic() - started))))
        current = read_health(args.url)
        if set(current.get("capture", {})) != camera_ids:
            issues.add("Camera roster changed during measurement")
        for key in camera_ids:
            if not current.get("capture", {}).get(key, {}).get("connected"):
                issues.add("{} disconnected during measurement".format(key))
            metric = current.get("cameras", {}).get(key, {})
            silence_ms = int(time.time() * 1000) - metric.get("last_result_ts_ms", 0)
            max_ages[key] = max(max_ages[key], metric.get("frame_age_ms", 0), silence_ms)

    elapsed = time.monotonic() - started
    rates = {}
    for key in sorted(camera_ids):
        before = baseline["cameras"][key]["infer_ok"]
        after = current.get("cameras", {}).get(key, {}).get("infer_ok", 0)
        rates[key] = round((after - before) / elapsed, 2)
        if rates[key] < args.fps * 0.9:
            issues.add("{} below 90% of target FPS".format(key))
        if max_ages[key] > args.max_age_ms:
            issues.add("{} exceeded frame-age/silence limit".format(key))
    counters = ("infer_fail", "infer_dropped", "pool_evicted_total", "pool_expired_total")
    deltas = {key: current[key] - baseline[key] for key in counters}
    if any(deltas.values()):
        issues.add("Failure/drop counters changed; reduce load or investigate")
    report = {"passed": not issues, "duration_s": round(elapsed, 2),
              "target_fps": args.fps, "per_camera_fps": rates,
              "max_sampled_age_or_silence_ms": max_ages,
              "counter_deltas": deltas, "issues": sorted(issues)}
    print(json.dumps(report, indent=2))
    return 1 if issues else 0


if __name__ == "__main__":
    raise SystemExit(main())
