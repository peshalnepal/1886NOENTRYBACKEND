from datetime import datetime, time as dt_time, timezone as dt_timezone
from typing import Optional, Literal, Tuple, Any, Dict, List
import uuid
from pydantic import BaseModel, ConfigDict, Field, model_validator
from zoneinfo import ZoneInfo

from dto import ChannelConfig

DAY_NAME_BY_VALUE = {
    0: "Monday",
    1: "Tuesday",
    2: "Wednesday",
    3: "Thursday",
    4: "Friday",
    5: "Saturday",
    6: "Sunday",
}
SUNDAY_TO_SATURDAY = [6, 0, 1, 2, 3, 4, 5]
DEFAULT_START_TIME = dt_time(0, 0, 0)
DEFAULT_END_TIME = dt_time(23, 59, 59)


def _coerce_schedule_time(value: Any, default: dt_time) -> dt_time:
    if value is None:
        return default
    if isinstance(value, dt_time):
        return value
    if isinstance(value, str):
        return dt_time.fromisoformat(value)
    raise ValueError(f"Unsupported schedule time value: {value!r}")


def _normalize_schedule_days(raw_days: Any) -> List[int]:
    values = raw_days if isinstance(raw_days, list) else [raw_days]
    normalized: List[int] = []
    seen = set()
    for value in values:
        try:
            day = int(value)
        except (TypeError, ValueError):
            continue
        if day < 0 or day > 6 or day in seen:
            continue
        seen.add(day)
        normalized.append(day)
    return normalized


class VideoChannelConfig(BaseModel, ChannelConfig):
    """
    One unified config for a Video Channel (create/edit/runtime).

    Rules:
    - CREATE (camera_uuid is None):
        requires rtsp_url, site_uuid, device_uuid
    - EDIT (camera_uuid is provided):
        all fields are optional; only provided ones should be applied.
    - webrtc_url and device_url:
        server-managed (read-only from client perspective). Keep them here so
        you can return them and pass them around internally.
    """
    model_config = ConfigDict(extra="forbid")
    camera_uuid: Optional[uuid.UUID] = Field(
        default=None,
        description="Camera UUID. Omit on create; include on edit."
    )
    channel_id: Optional[str] = Field(
        default=None,
        description="Runtime channel key. If omitted, use str(camera_uuid)."
    )

    site_uuid: Optional[uuid.UUID] = Field(
        default=None,
        description="Owning site UUID (required on create)."
    )

    device_uuid: Optional[uuid.UUID] = Field(
        default=None,
        description="Assigned device UUID (required on create)."
    )

    rtsp_url: Optional[str] = Field(
        default=None,
    )

    webrtc_url: Optional[str] = Field(
        default=None,
    )

    device_url: Optional[str] = Field(
        default=None,
        description="Server-managed base URL of assigned Jetson/device.",
        json_schema_extra={"readOnly": True},
    )
    name: Optional[str] = None
    location: Optional[str] = None
    timezone: Optional[str] = None
    schedule: Optional[List[Dict[str, Any]]] = None
    use_site_schedule: Optional[bool] = None
    enabled: Optional[bool] = Field(default=True)
    detection_enabled: Optional[bool] = Field(default=True)
    notification_enabled: Optional[bool] = Field(default=True)
    sample_fps: Optional[float] = Field(default=5.0, ge=0.1)
    decode_backend: Optional[Literal["gstreamer", "opencv"]] = Field(default="gstreamer")
    resize: Optional[Tuple[int, int]] = Field(default=None, description="(width, height)")
    reconnect_base_ms: Optional[int] = Field(default=1000, ge=100)
    reconnect_max_ms: Optional[int] = Field(default=8000, ge=1000)
    poll_interval_ms: Optional[int] = Field(default=200, ge=10)
    request_timeout_s: Optional[float] = Field(default=3.0, ge=0.1)
    detection_path_template: Optional[str] = Field(
        default="/detection/{camera_uuid}",
        description="Jetson detection endpoint template.",
    )
    emit_format: Optional[Literal["raw", "jpeg"]] = Field(
        default=None,
        description="DEBUG ONLY. Backend should not stream frames in production."
    )
    jpeg_quality: Optional[int] = Field(default=None, ge=1, le=100)

    @staticmethod
    def default_schedule() -> List[Dict[str, Any]]:
        return [
            {
                "day_of_week": day,
                "day_name": DAY_NAME_BY_VALUE[day],
                "start_time": DEFAULT_START_TIME.strftime("%H:%M:%S"),
                "end_time": DEFAULT_END_TIME.strftime("%H:%M:%S"),
                "is_enabled": True,
            }
            for day in SUNDAY_TO_SATURDAY
        ]

    @staticmethod
    def normalize_schedule(raw_schedule: Any) -> List[Dict[str, Any]]:
        if not isinstance(raw_schedule, list):
            return []

        normalized: List[Dict[str, Any]] = []
        seen = set()
        order_index = {day: idx for idx, day in enumerate(SUNDAY_TO_SATURDAY)}

        day_start = dt_time(0, 0, 0)
        day_end = dt_time(23, 59, 59)

        def add_window(day: int, start_time: dt_time, end_time: dt_time, enabled: bool) -> None:
            if start_time == end_time:
                return
            signature = (day, start_time.isoformat(), end_time.isoformat(), enabled)
            if signature in seen:
                return
            seen.add(signature)

            normalized.append(
                {
                    "day_of_week": day,
                    "day_name": DAY_NAME_BY_VALUE[day],
                    "start_time": start_time.strftime("%H:%M:%S"),
                    "end_time": end_time.strftime("%H:%M:%S"),
                    "is_enabled": enabled,
                }
            )

        for item in raw_schedule:
            if not isinstance(item, dict):
                continue

            days = _normalize_schedule_days(item.get("day_of_week"))
            if not days:
                continue

            try:
                start_time = _coerce_schedule_time(
                    item.get("start_time"), DEFAULT_START_TIME
                )
                end_time = _coerce_schedule_time(
                    item.get("end_time"), DEFAULT_END_TIME
                )
            except ValueError:
                continue

            enabled = bool(item.get("is_enabled", True))

            for day in days:
                # Normal same-day window
                if start_time < end_time:
                    add_window(day, start_time, end_time, enabled)
                    continue

                # Overnight window, e.g. 18:00:00 -> 06:00:00
                next_day = (day + 1) % 7
                add_window(day, start_time, day_end, enabled)
                add_window(next_day, day_start, end_time, enabled)

        return sorted(
            normalized,
            key=lambda item: (
                order_index.get(int(item["day_of_week"]), 999),
                item["start_time"],
            ),
        )

    @staticmethod
    def schedule_windows(raw_schedule: Any) -> List[Dict[str, Any]]:
        schedule = VideoChannelConfig.normalize_schedule(raw_schedule)
        if not schedule:
            return []

        visible = [entry for entry in schedule if bool(entry.get("is_enabled", True))]
        if not visible:
            visible = list(schedule)

        parsed: List[Dict[str, Any]] = []
        for entry in visible:
            if not isinstance(entry, dict):
                continue
            days = _normalize_schedule_days(entry.get("day_of_week"))
            if not days:
                continue
            try:
                start_time = _coerce_schedule_time(
                    entry.get("start_time"), DEFAULT_START_TIME
                )
                end_time = _coerce_schedule_time(
                    entry.get("end_time"), DEFAULT_END_TIME
                )
            except ValueError:
                continue
            if start_time == end_time:
                continue
            for day in days:
                parsed.append(
                    {
                        "day_of_week": day,
                        "start_time": start_time,
                        "end_time": end_time,
                        "is_enabled": bool(entry.get("is_enabled", True)),
                    }
                )

        if not parsed:
            return []

        paired_target_indexes = set()
        overnight_pairs: Dict[int, int] = {}
        for idx, entry in enumerate(parsed):
            if entry["start_time"] <= DEFAULT_START_TIME:
                continue
            if entry["end_time"] != DEFAULT_END_TIME:
                continue

            next_day = (int(entry["day_of_week"]) + 1) % 7
            for target_idx, target in enumerate(parsed):
                if target_idx == idx or target_idx in paired_target_indexes:
                    continue
                if int(target["day_of_week"]) != next_day:
                    continue
                if target["start_time"] != DEFAULT_START_TIME:
                    continue
                if target["end_time"] >= DEFAULT_END_TIME:
                    continue
                if bool(target["is_enabled"]) != bool(entry["is_enabled"]):
                    continue
                overnight_pairs[idx] = target_idx
                paired_target_indexes.add(target_idx)
                break

        grouped: Dict[Tuple[str, str, bool], Dict[str, Any]] = {}
        order_index = {day: idx for idx, day in enumerate(SUNDAY_TO_SATURDAY)}

        for idx, entry in enumerate(parsed):
            if idx in paired_target_indexes:
                continue

            end_time = entry["end_time"]
            if idx in overnight_pairs:
                end_time = parsed[overnight_pairs[idx]]["end_time"]

            start_str = entry["start_time"].strftime("%H:%M:%S")
            end_str = end_time.strftime("%H:%M:%S")
            enabled = bool(entry["is_enabled"])
            key = (start_str, end_str, enabled)
            grouped.setdefault(
                key,
                {
                    "day_of_week": [],
                    "start_time": start_str,
                    "end_time": end_str,
                    "is_enabled": enabled,
                },
            )
            day = int(entry["day_of_week"])
            if day not in grouped[key]["day_of_week"]:
                grouped[key]["day_of_week"].append(day)

        collapsed = list(grouped.values())
        for entry in collapsed:
            entry["day_of_week"] = sorted(
                entry["day_of_week"],
                key=lambda day: order_index.get(int(day), 999),
            )

        return sorted(
            collapsed,
            key=lambda entry: (
                order_index.get(int((entry.get("day_of_week") or [999])[0]), 999),
                entry.get("start_time") or "",
                entry.get("end_time") or "",
            ),
        )
    @staticmethod
    def schedule_is_active(
        raw_schedule: Any,
        timezone_name: Optional[str] = None,
        *,
        now_utc: Optional[datetime] = None,
    ) -> bool:
        schedule = VideoChannelConfig.normalize_schedule(raw_schedule)
        if not schedule:
            return True

        now = now_utc or datetime.now(dt_timezone.utc)
        if now.tzinfo is None:
            now = now.replace(tzinfo=dt_timezone.utc)

        try:
            tz = ZoneInfo(str(timezone_name or "UTC"))
        except Exception:
            tz = dt_timezone.utc

        local_now = now.astimezone(tz)
        local_day = int(local_now.weekday())
        local_time = local_now.time()

        for window in schedule:
            if not bool(window.get("is_enabled", True)):
                continue
            try:
                window_day = int(window.get("day_of_week", -1))
            except (TypeError, ValueError):
                continue

            if window_day != local_day:
                continue

            try:
                start_time = _coerce_schedule_time(
                    window.get("start_time"), DEFAULT_START_TIME
                )
                end_time = _coerce_schedule_time(
                    window.get("end_time"), DEFAULT_END_TIME
                )
            except ValueError:
                continue

            if start_time == end_time:
                continue
            if start_time <= local_time < end_time:
                return True

        return False

    def is_scheduled_now(self, *, now_utc: Optional[datetime] = None) -> bool:
        return self.schedule_is_active(self.schedule, self.timezone, now_utc=now_utc)

    @model_validator(mode="after")
    def _validate_and_normalize(self):
        for attr in (
            "channel_id",
            "rtsp_url",
            "webrtc_url",
            "device_url",
            "name",
            "location",
            "timezone",
            "detection_path_template",
        ):
            v = getattr(self, attr, None)
            if isinstance(v, str) and not v.strip():
                setattr(self, attr, None)

        if self.channel_id is None and self.camera_uuid is not None:
            self.channel_id = str(self.camera_uuid)

        if self.schedule is not None:
            self.schedule = self.normalize_schedule(self.schedule) or None

        if self.camera_uuid is None:
            if not self.rtsp_url:
                raise ValueError("rtsp_url is required when creating a new camera (camera_uuid is None).")
            if self.site_uuid is None:
                raise ValueError("site_uuid is required when creating a new camera.")
            if self.device_uuid is None:
                raise ValueError("device_uuid is required when creating a new camera.")

        return self

    # -----------------------------
    # Helpers (optional but useful)
    # -----------------------------
    def to_patch_dict(self) -> Dict[str, Any]:
        """
        For edit operations: returns only fields that were actually provided by the client,
        excluding read-only server-managed fields.
        """
        d = self.model_dump(exclude_none=True, exclude_unset=True)
        d.pop("webrtc_url", None)
        d.pop("device_url", None)
        return d
