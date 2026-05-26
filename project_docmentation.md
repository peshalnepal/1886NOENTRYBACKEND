# 1886NOENTRY Backend — Project Documentation

> Comprehensive technical breakdown of the FastAPI backend (excluding `tensort/` and `tensorrt_test/`).
> Traces every layer (entry point → core → domain → application → routes), every key class, and the end-to-end flows that tie them together. Up to date as of the 2026-05-11 refactor pass ([Section 10](#10-refactoring-notes--history)).

---

## Table of Contents

1. [High-Level Architecture](#1-high-level-architecture)
2. [Entry Point & App Bootstrap](#2-entry-point--app-bootstrap)
3. [Core Layer](#3-core-layer)
4. [Domain Layer](#4-domain-layer)
5. [Application Layer (Services / Repositories / Channels / Models / Builders)](#5-application-layer)
6. [Routes Layer (HTTP API)](#6-routes-layer-http-api)
7. [Scripts, Deployment & Tests](#7-scripts-deployment--tests)
8. [End-to-End Flows](#8-end-to-end-flows)
9. [Cross-Cutting Concerns](#9-cross-cutting-concerns)
10. [Refactoring Notes & History](#10-refactoring-notes--history)

---

## 1. High-Level Architecture

This backend powers a **multi-tenant AI video-surveillance platform**. Users connect cameras (RTSP streams) to Jetson edge devices that run TensorRT YOLO inference, while the FastAPI server orchestrates state, persists detections/clips, and pushes real-time notifications to the dashboard.

**Stack**
- **API**: FastAPI (async)
- **DB**: SQLAlchemy 2.x async (aiomysql for MySQL; aioodbc for MSSQL/Postgres also supported)
- **Auth**: JWT (HS256) + bcrypt + HMAC-SHA256 OTP
- **Edge**: HTTP client to Jetson TensorRT API (see [tensort/](tensort/) folder — documented separately)
- **WebRTC**: MediaMTX gateway for RTSP→WebRTC (WHEP) playback
- **Storage**: Azure Blob Storage for alert JPEGs and video clips
- **Background**: asyncio loops for edge reconciliation, retention cleanup
- **Realtime**: SSE / in-memory hub for notifications

**Layering**

```
HTTP request
   │
   ▼
routes/*           ── FastAPI routers, request/response shaping, auth dep
   │
   ▼
application/services/*  ── business logic, orchestration (Manager, Notification, …)
   │
   ▼
application/repositories/*  ── DB access (does NOT commit; caller owns txn)
   │
   ▼
core/database_orm.py   ── SQLAlchemy 2.x ORM models
   │
   ▼
core/database.py       ── DatabaseManager (engine, sessions, migrations)
```

Cross-cutting:
- **domain/** — pure dataclasses / Pydantic events used as transport between layers
- **application/channels/** & **application/models/** — pipeline building blocks (video channels, YOLO model adapter)
- **dependencies.py** — FastAPI `Depends` factories
- **Shared helpers** — [core/env.py](core/env.py) (env-var parsers), [application/services/overlay_normalize.py](application/services/overlay_normalize.py) (overlay/detection normalization, used by domain, services, and routes), [routes/_background.py](routes/_background.py) (bg-task + blob-cleanup helpers)

---

## 2. Entry Point & App Bootstrap

### [main.py](main.py)

**Lifespan ([main.py L128–L237](main.py#L128-L237))**

1. Calls `db_manager.initialize_tables_and_data()` ([main.py L138-L142](main.py#L138-L142)) which:
   - Creates all ORM tables (`Base.metadata.create_all`)
   - Runs idempotent schema-patch migrations (column additions, index creation, soft-delete backfills)
   - Seeds a dev user (`dev@example.com` / `DevPass123!`) on a fresh DB
2. Constructs the **`Manager` singleton** (pipeline & device orchestration) and stores it on `app.state.manager`.
3. Constructs **`NotificationService`** wired to a **`WebNotificationHub`** (in-memory SSE/WS broadcaster).
4. Configures **`EmailNotifier`** from SMTP env vars.
5. Constructs **`RetentionService`** (TTL cleanup for clips/alerts).
6. Spawns two background asyncio tasks:
   - `_edge_reconcile_loop` ([main.py L50-L102](main.py#L50-L102)) — every `EDGE_RECONCILE_INTERVAL_S` (default 30 s) calls `Manager.reconcile_all_devices_edge()`, with exponential backoff on failure.
   - `_retention_cleanup_loop` ([main.py L104-L126](main.py#L104-L126)) — every `RETENTION_CLEANUP_INTERVAL_S` (default 3600 s) calls `RetentionService.purge()`.
7. On shutdown, cancels both loops and awaits queued blob-cleanup tasks (`app.state.alert_blob_cleanup_tasks`).

**Middleware ([main.py L241-L257](main.py#L241-L257))**
- `Cache-Control: no-cache, no-store` on every response.
- CORS: allow-all origins (no credentials).

**Routers mounted at `/api` ([main.py L261-L268](main.py#L261-L268))**

| Prefix | Module |
|---|---|
| `/api/cameras` | [routes/camera_routes.py](routes/camera_routes.py) |
| `/api/clips` | [routes/clips_routes.py](routes/clips_routes.py) |
| `/api/sites` | [routes/site_routes.py](routes/site_routes.py) |
| `/api/devices` | [routes/device_routes.py](routes/device_routes.py) |
| `/api/notifications` | [routes/notifications_routes.py](routes/notifications_routes.py) |
| `/api/notification-emails` | [routes/notification_email_routes.py](routes/notification_email_routes.py) |
| `/api/auth` | [routes/auth/__init__.py](routes/auth/__init__.py) |
| `/api/users` | [routes/user_routes.py](routes/user_routes.py) |

### [dependencies.py](dependencies.py) — FastAPI DI

| Provider | Returns | Purpose |
|---|---|---|
| `get_async_db` | `AsyncSession` | Per-request DB session; closes on exit |
| `get_current_user` | `User` ORM | Decodes Bearer JWT; raises 401 on failure |
| `get_manager` | `Manager` | App-state singleton; 503 if not initialised |
| `get_session_factory` | `async_sessionmaker[AsyncSession]` | For services that need to scope their own transactions |
| `get_notification_service` | `NotificationService` | Emit detections/alerts |
| `get_notification_hub` | `WebNotificationHub` | SSE broadcast |
| `get_user_snapshot_cache` | `UserSnapshotCache` | Cached user metadata for streams |
| `get_retention_service` | `RetentionService` | Manual purge endpoints |

### Other top-level files

There are no loose `.py` files at `Backend/` root other than [main.py](main.py) and [dependencies.py](dependencies.py). Earlier the root also held `utils.py`, `dto.py`, `verify_fix.py`, and an `Updated/` folder of abandoned drafts — these were all removed during the refactor (see [Section 10](#10-refactoring-notes--history)). DTOs are now split into the folders that own them ([application/models/vision_config.py](application/models/vision_config.py), [application/channels/channel_config.py](application/channels/channel_config.py)).

---

## 3. Core Layer

### [core/config.py](core/config.py)

Loads `.env` via `python-dotenv` and exports module-level constants: `DEBUG`, `DATABASE_URL`, SMTP credentials, etc. There is no settings class — everything is read directly from `os.environ`. This is something a refactor would likely consolidate into a `pydantic_settings.BaseSettings` model.

### [core/database.py](core/database.py) — `DatabaseManager`

A cross-database abstraction supporting MySQL (aiomysql/pymysql), MSSQL (aioodbc/pyodbc), and PostgreSQL.

- `_setup()` parses the ODBC-style connection string and dispatches to `_setup_mysql()` / `_setup_mssql()`.
- Builds **both** a sync engine (`engine`, `SessionLocal`) and an async engine (`async_engine`, `AsyncSessionLocal`). Async is the primary path; sync survives in a few legacy helpers.
- Pool tuning via env: `DB_POOL_SIZE` (30), `DB_MAX_OVERFLOW` (20), `DB_POOL_TIMEOUT_S` (30), `DB_POOL_RECYCLE_S` (1800), `DB_CONNECT_TIMEOUT_S` (5), `DB_IO_TIMEOUT_S` (15).
- **`initialize_tables_and_data()` ([core/database.py L192-L569](core/database.py#L192-L569))** is the migration entry point. It:
  1. Runs `Base.metadata.create_all()`.
  2. Applies hand-written idempotent migrations against `information_schema.COLUMNS`/`STATISTICS`:
     - `email_verifications.code` → nullable, `code_hash` added (OTPs are hashed).
     - `video_record.overlay_payload` JSON column.
     - `site_settings` schedule columns (`config`, `day_of_week`, `start_time`, `end_time`, `is_enabled`).
     - `notification.visible` soft-delete flag + backfill.
     - Composite indexes on `notification` and `video_record` for paginated lookups.
     - `sites.is_deleted` soft-delete flag.
     - Tri-state enums: `camera.notification_trigger_mode`, `camera.camera_playback_enabled`.
  3. Seeds the dev user if `users` is empty.

> ⚠️ `create_all` never alters existing tables. New schema changes to existing tables must be added as explicit migration blocks here.

### [core/database_orm.py](core/database_orm.py) — ORM models

**Custom helpers**
- `GUIDType` ([L37-L78](core/database_orm.py#L37-L78)) — portable UUID column (BINARY(16) on MySQL, UNIQUEIDENTIFIER on MSSQL, UUID on PG, CHAR(36) elsewhere).
- `DeepMutableDict` / `JSONDict` / `JSONList` ([L88-L139](core/database_orm.py#L88-L139)) — Mutable-tracked JSON columns so in-place mutations are flushed.
- `utc_now()` — single definition (was previously duplicated; cleaned up).

**Tables**

| Table | Key fields | Relationships |
|---|---|---|
| `User` ([L154-L186](core/database_orm.py#L154-L186)) | id, user_name, email (uniq), hashed_password, email_verified | sites, devices, cameras, notifications, notification_emails (all cascade) |
| `Site` ([L191-L248](core/database_orm.py#L191-L248)) | site_uuid PK, user_id FK, site_code, name, timezone, is_deleted | user; cameras; devices (M:N); settings 1:1; notification_emails |
| `Device` ([L254-L297](core/database_orm.py#L254-L297)) | device_uuid PK, user_id, device_code, device_url, is_enabled | user; sites M:N; cameras M:N |
| `Camera` ([L362-L425](core/database_orm.py#L362-L425)) | camera_uuid, site_uuid, rtsp_url, webrtc_url, roi (JSON), tri-state mode flags | site; devices M:N; channel_configuration 1:1; video_records; pipelines M:N |
| `ChannelConfiguration` ([L493-L522](core/database_orm.py#L493-L522)) | runtime VideoChannelConfig JSON + schedule (day_of_week, start/end time) | camera 1:1 |
| `Pipeline` ([L452-L471](core/database_orm.py#L452-L471)) | UUID PK, user_id, name, is_active | cameras M:N |
| `VideoRecord` ([L528-L558](core/database_orm.py#L528-L558)) | clip metadata: camera_uuid FK, start/end_time, storage_key, overlay_payload JSON, status | camera |
| `Notification` ([L561-L599](core/database_orm.py#L561-L599)) | user_id, site_uuid, camera_uuid?, device_uuid?, payload JSON, detected_at, read_at, visible (soft) | user, site, camera, device |
| `EmailVerification`, `SignupTempData` | OTP storage with `code_hash`, attempts, expires_at | — |
| `SiteSettings`, `NotificationEmail` | per-site config and email recipients | site |

Indexes are tuned for the most common queries (notification list by user+visible+detected_at DESC; clips by camera+created_at).

### [core/schemas.py](core/schemas.py) — Pydantic

- `ROISchema` ([L15-L23](core/schemas.py#L15-L23)) — polygon vertices + frame dims.
- `CameraBaseSchema` ([L85-L131](core/schemas.py#L85-L131)) — common fields for create/edit including tri-state inheritance flags.
- `CameraCreateSchema` ([L133-L192](core/schemas.py#L133-L192)) — required-field POST shape.
- `CameraEditSchema` ([L199-L250](core/schemas.py#L199-L250)) — all-optional PATCH shape.
- Auth/user/site/device/notification/clip request and response shapes also live here.

### [core/security/hashing.py](core/security/hashing.py)
- `get_password_hash(plain)` / `verify_password(plain, hashed)` via `passlib.CryptContext` (bcrypt).
- `generate_random_key()` — URL-safe 32-byte token (used for OTP material / API keys).
- `hash_key(key)` — SHA-256 hex for indexed lookup.

### [core/security/tokens.py](core/security/tokens.py)
- `create_access_token(data)` — HS256 JWT signed with `SECRET_KEY`, 24 h expiry by default.
- `decode_access_token(token)` — validates signature + expiry; raises `ValueError` on failure (translated to 401 by `get_current_user`).

### [core/env.py](core/env.py) — env-var parsers

Three small typed wrappers around `os.getenv`, used in place of duplicated parsers that previously lived in `main.py`, `routes/auth/signup.py`, `core/database.py`, `application/services/edgeinference.py`, `application/services/notification/*`, `application/repositories/verify_repository.py`, and `application/channels/channel.py`.

- `env_bool(name, default=False)` — accepts `1/true/yes/on` (case-insensitive).
- `env_int(name, default, *, minimum=None)` — coerces to int with optional floor.
- `env_float(name, default, *, minimum=None)` — same for floats.

> `core/utils.py` previously sat empty in the tree and has been removed.

---

## 4. Domain Layer

Pure data-shape definitions (Pydantic / dataclasses). No I/O, no DB.

### [domain/events.py](domain/events.py)

| Event | Purpose |
|---|---|
| `Event` (base) | event_type + payload dict |
| `ChannelEvent` | Base for anything sourced from a channel |
| `RTSPEvent` ([L42-L78](domain/events.py#L42-L78)) | Frame envelope: camera_uuid, ts_ms, seq, format (raw/jpeg/h264), width/height, fps_hint, keyframe, detection_enabled |
| `ChannelConnectedEvent` / `ChannelDisconnectedEvent` | Stream lifecycle |
| `ChannelCreateEvent` / `ChannelRemoveEvent` / `ChannelEditEvent` | Admin ops broadcast through the pipeline |
| `DetectionsProducedEvent` ([L180-L191](domain/events.py#L180-L191)) | Inference output: detections + pose + inference_ms |
| `DetectionItem`, `DetectionBox`, `PoseResult`, `SkeletonItem`, `PoseKeypoint` | Detection payload structure |
| `AlertRaisedEvent` | ROI/zone triggered by a tracked detection |
| `ClipRequestEvent` / `ClipReadyEvent` | Pre-record clip flow |

### [domain/model.py](domain/model.py)
`VisionModel` Protocol — `async infer(rtsp_ev) -> ChannelEvent`. Implementations live in [application/models/](application/models/).

### [domain/model_pipeline.py](domain/model_pipeline.py)
`ObjDetectResponse` dataclass ([L39-L62](domain/model_pipeline.py#L39-L62)) — uniform shape returned from polling/streaming detection (detections, pose, tracks, alerts, inference_ms, error reason).

### [domain/channel.py](domain/channel.py)
`Channel` abstract base — bidirectional event stream contract, implemented by `VideoChannel`.

### [domain/template.py](domain/template.py)
`Template` — pipeline blueprint: id + list of channel configs + optional model_cfg. Consumed by `PipelineBuilder`.

---

## 5. Application Layer

### 5.1 Repositories — `application/repositories/`

All repository methods take an `AsyncSession` and **do not commit**. The caller (route or service) owns the transaction boundary.

#### [user_repository.py](application/repositories/user_repository.py)
- `get_by_id(db, user_id)`
- `get_by_email(db, email)`
- `exists_email(db, email)`
- `create_user(db, ...)` — raises `ValueError` on duplicate email.
- `mark_email_verified(db, user)`
- `update_last_login(db, user)`
- `update_password_hash(db, user, new_hash)`
- `update_profile(db, user, **fields)`

#### [channel_repository.py](application/repositories/channel_repository.py)
- `upsert_camera_from_channel_config(...)` — the core write path. Creates/updates `Camera`, `ChannelConfiguration`, and the `PipelineCamera` link in one go. Validates the pipeline exists; normalises name/location/schedule/timezone.
- `get_camera_full(...)` — eager-loads camera + channel_configuration + devices.
- `get_associated_devices(...)` — returns list of devices linked through the M:N association (exactly one is expected).

#### [site_repository.py](application/repositories/site_repository.py)
- `list_cameras_by_site(...)`
- `list_devices_for_site(...)`
- `get_site_settings(...)` / `upsert_site_settings(...)` — schedule + prerecord config.

#### [pipeline_repository.py](application/repositories/pipeline_repository.py)
- `pipeline_exists`, `upsert_pipeline` (with `IntegrityError` retry for race), `list_pipelines`, `add_camera_to_pipeline`, `remove_camera_from_pipeline`.

#### [notification_repository.py](application/repositories/notification_repository.py)
- `CameraContext` dataclass — denormalised lookup result used to compose notifications without N+1 queries.
- `get_camera_context(...)` — single query to assemble user/site/camera/device + tri-state modes.
- `get_site_prerecord_settings(...)` — prerecord trigger + enabled camera list.
- `list_notification_emails(...)`
- `insert_notification(...)` / `batch_insert_notifications(...)`
- `mark_read`, `mark_unread`, `delete_by_id`.
- Helpers: `_coerce_trigger_mode`, `_normalize_uuid_list`.

#### [notification_visibility.py](application/repositories/notification_visibility.py)
Soft-delete primitives (set `visible=False` rather than DELETE).

#### [verify_repository.py](application/repositories/verify_repository.py)
Email-verification OTP storage. Each bulk `update()` must include `.execution_options(synchronize_session=False)` — see CLAUDE.md for the historical bug context.

### 5.2 Services — `application/services/`

#### [manager/](application/services/manager/) — `Manager` (package)

The orchestration hub — one singleton per process. Previously a single 2074-line `manager.py`; now split into a package whose public surface is unchanged. External code still does `from application.services.manager import Manager, EdgeDeviceUnavailableError`.

Public re-exports live in [manager/__init__.py](application/services/manager/__init__.py): `Manager`, `EdgeDeviceUnavailableError`, `CameraOut`, `PipelineUpdateResult`.

**Layout**

| File | Lines | What it holds |
|---|---:|---|
| [`__init__.py`](application/services/manager/__init__.py) | 23 | Re-exports |
| [`types.py`](application/services/manager/types.py) | 61 | `CameraOut`, `PipelineUpdateResult`, `EdgeDeviceUnavailableError` |
| [`helpers.py`](application/services/manager/helpers.py) | 128 | Module-level constants + helpers (`_edge_health_ready`, `_only_jetson_config`, `_runtime_config_overrides`, `_coerce_tri_*`, `_edge_runtime_enabled`, `HARD_PATCH_KEYS`, `JETSON_PATCH_KEYS`, `RUNTIME_CONFIG_*_KEYS`) |
| [`service.py`](application/services/manager/service.py) | 245 | `Manager` class declaration (joins all mixins) + `__init__` + `shutdown` + `start_background_pipelines` + `set_notification_service` + `_wire_pipeline` + `_event_includes_roi_patch` + `_invalidate_camera_roi_state` + `_get_user_lock` + `_call_with_timeout` + `_patch_to_dict` + `_as_uuid` |
| [`_service_devices.py`](application/services/manager/_service_devices.py) | 283 | `ManagerDevicesMixin` — `_get_device`, `_get_site_devices`, `_pick_site_device_uuid`, `_resolve_site_device`, `_ensure_site_owned_by_user`, `_get_single_camera_device`, `_get_camera_devices`, `_normalize_device_url`, `_list_devices_for_physical_device`, `_list_cameras_for_device_uuids`, `_list_enabled_reconcile_devices`, `_set_single_camera_device`, `_edge_payload_from_config`, `_get_site` |
| [`_service_schedule.py`](application/services/manager/_service_schedule.py) | 262 | `ManagerScheduleMixin` — `_load_site_schedule_state`, `_get_loaded_channel_configuration`, `_resolve_runtime_schedule`, `sync_site_schedule_runtime` |
| [`_service_reconcile.py`](application/services/manager/_service_reconcile.py) | 417 | `ManagerReconcileMixin` — `reconcile_devices_best_effort`, `_reconcile_device_edge_with_retry`, `reconcile_device_edge_simple`, `reconcile_all_devices_edge` |
| [`_service_pipeline.py`](application/services/manager/_service_pipeline.py) | 344 | `ManagerPipelineMixin` — `_create_pipeline_unlocked`, `create_pipeline`, `get_activepipeline`, `get_loaded_pipeline`, `update_pipeline`, `_bg_edge_upsert`, `_bg_edge_delete`, `_bg_edge_patch`, `_bg_webrtc_update_stream` |
| [`_service_channel.py`](application/services/manager/_service_channel.py) | 498 | `ManagerChannelMixin` — `_add_channel`, `_edit_channel`, `_remove_channel` (the camera CRUD path) |
| [`_service_cleanup.py`](application/services/manager/_service_cleanup.py) | 205 | `ManagerCleanupMixin` — `cleanup_user_resources`, `cleanup_device_resources`, `cleanup_site_resources` |

**Class composition** — [service.py](application/services/manager/service.py) declares:

```python
class Manager(
    ManagerChannelMixin,
    ManagerCleanupMixin,
    ManagerDevicesMixin,
    ManagerPipelineMixin,
    ManagerReconcileMixin,
    ManagerScheduleMixin,
):
    def __init__(self, session_factory): ...
```

All mixins read/write `self._*` state set up by `__init__` in `service.py`. They are not standalone — only valid as bases of `Manager`.

**Behavior** (unchanged from the pre-split monolith):

- `start_background_pipelines()` — load enabled pipelines from DB, instantiate `ModelPipeline` + `VideoChannel` per pipeline, spawn asyncio tasks.
- `get_loaded_pipeline(user_id)` — fetch in-memory `ModelPipeline`.
- `_add_channel(...)` (camera-create path) — `Camera` + `ChannelConfiguration` upsert → pipeline registration → `EdgeInferenceClient.add_camera()` → `WebRTCGatewayClient.provision_stream()`.
- `_edit_channel(...)` — patch semantics; revalidates device; may trigger hard-reset on structural change.
- `_remove_channel(...)` — pipeline removal + edge delete + WebRTC unprovision.
- `cleanup_device_resources(...)` — when a device is deleted, iterate its cameras and tear them down.
- `reconcile_all_devices_edge()` — polls each edge device's `/cameras` inventory, compares with DB, optionally deletes unknowns (`EDGE_RECONCILE_DELETE_UNKNOWN`).
- `shutdown()` — cancel tasks, close clients.
- Custom error: `EdgeDeviceUnavailableError(device_url, cause)`.

> **Finding a method**: each mixin file's docstring lists which methods live there. A quick `grep -rn "def <method>" application/services/manager/` will jump straight to the owning file.

#### [notification/](application/services/notification/) — `NotificationService` + `WebNotificationHub` + `EmailNotifier` (package)

Previously a single 2821-line `notification.py`; now split into a package whose public surface is unchanged. External code still does `from application.services.notification import NotificationService, WebNotificationHub, EmailNotifier, EmailConfig, NotificationMessage`.

Public re-exports live in [notification/__init__.py](application/services/notification/__init__.py).

**Layout**

| File | Lines | What it holds |
|---|---:|---|
| [`__init__.py`](application/services/notification/__init__.py) | 33 | Re-exports |
| [`types.py`](application/services/notification/types.py) | 90 | `NotificationMessage` (BaseModel), `BufferedNotification`, `BufferedDeletion`, `CameraMode`, `SitePrerecordPlan`, `EmailConfig` |
| [`overlay_helpers.py`](application/services/notification/overlay_helpers.py) | 240 | Frame/overlay normalization: `_json_safe`, `_coerce_positive_int`, `_normalize_overlay_frame`, `_merge_overlay_frames`, `_select_overlay_reference_frame`, `_build_overlay_payload_from_frames`, `_parse_utc_datetime`, `_event_overlay_payload` |
| [`hub.py`](application/services/notification/hub.py) | 57 | `WebNotificationHub` — in-process subscriber registry; `subscribe(user_id)` returns an `asyncio.Queue`; `publish(msg)` fans out (dropping oldest on QueueFull). |
| [`email_notifier.py`](application/services/notification/email_notifier.py) | 345 | `EmailNotifier` — `send`, `send_digest`, plus `_render_text` / `_render_html` / `_render_digest_text` / `_render_digest_html`. SMTP via `smtplib` on a thread. |
| [`service.py`](application/services/notification/service.py) | 292 | `NotificationService` class declaration (joins all mixins) + `__init__` + `set_session_factory` + `shutdown` + `purge_deleted_site_runtime_state` + long-running task plumbing (`_fire_and_forget`, `_ensure_delete_task`, `_ensure_flush_task`, `_delete_loop`, `_flush_loop`, `_flatten_user_alerts_preserving_order`) |
| [`_service_deletion.py`](application/services/notification/_service_deletion.py) | 336 | `NotificationServiceDeletionMixin` — `_delete_alert_blob_keys`, `_batch_hide_notifications`, `_flush_delete_queue`, `handle_deletion_event` |
| [`_service_clip.py`](application/services/notification/_service_clip.py) | 376 | `NotificationServiceClipMixin` — `record_detection_overlay_frame`, `_build_clip_overlay_payload`, `_clip_overlay_frames_for_window`, `_trim_clip_overlay_history`, `_finalize_captured_clip`, `_attach_clip_payload` |
| [`_service_prerecord.py`](application/services/notification/_service_prerecord.py) | 341 | `NotificationServicePrerecordMixin` — `is_camera_prerecord_eligible`, `invalidate_prerecord_eligible_cache`, `_materialize_alert_image_payload`, `_site_prerecord_clip_payload`, `_site_prerecord_trigger_matches`, `_load_site_prerecord_plan`, `_capture_site_prerecord_clips` |
| [`_service_flush.py`](application/services/notification/_service_flush.py) | 564 | `NotificationServiceFlushMixin` — `invalidate_recipient_cache`, `_flush_ready_users`, `_get_recipients_for_sites_cached`, `_flush_user_batch`, `_get_camera_ctx_cached`, `_prepare_notification_item`, `_persist_notification_now`, `enqueue_notification`, `_persist_and_send` |
| [`_service_detection.py`](application/services/notification/_service_detection.py) | 552 | `NotificationServiceDetectionMixin` — ROI parsing (`_parse_roi_points`, `_coerce_roi_normalized`, `_parse_roi_frame_size`, `_clamp_unit_points`), trigger-mode cache (`_get_site_trigger_mode`, `invalidate_site_trigger_mode_cache`, `invalidate_camera_roi_state`), ROI cache (`_get_rois`), and `handle_detection_event` + `_extract_interesting` |

**Class composition** — [service.py](application/services/notification/service.py) declares:

```python
class NotificationService(
    NotificationServiceClipMixin,
    NotificationServicePrerecordMixin,
    NotificationServiceFlushMixin,
    NotificationServiceDetectionMixin,
    NotificationServiceDeletionMixin,
):
    def __init__(self, *, hub, email=None, ...): ...
```

**Behavior** (unchanged from the pre-split monolith):

- `WebNotificationHub.subscribe(user_id)` returns an `asyncio.Queue[NotificationMessage]`; `publish(msg)` fan-outs to every queue subscribed for that user. Queue overflow drops the oldest item.
- `EmailNotifier.send(...)` / `send_digest(...)` — `MIMEMultipart` over `smtplib` on a thread. Config: `enabled`, host/port/user/pass, from address, dashboard base URL, subject prefix, camera URL template.
- `NotificationService.handle_detection_event(...)` — the main detection sink (in [`_service_detection.py`](application/services/notification/_service_detection.py)). Runs **ByteTrack** + **ROIAlertEngine** from [tracker.py](application/services/tracker.py) to derive confirmed tracks and alert events, then `_prepare_notification_item()` denormalises into a row, `_persist_notification_now()` flushes + commits with rollback on failure, and `_flush_user_batch()` batch-inserts.
- `NotificationService.handle_deletion_event(...)` — paired with detection events; soft-hides notifications when a user dismisses an alert, batching the DB updates with `_batch_hide_notifications` and `_flush_delete_queue`.
- `NotificationService.record_detection_overlay_frame(...)` — buffers per-camera overlay frames so that when a captured clip lands, `_finalize_captured_clip` can attach the matching detection overlays from the clip's [start, end] window.
- `notify_on_confirmed` flag — when set, emit `item_detected` once a track is confirmed even without ROI intersection (useful for prerecord-on-any-detection mode).
- Side effects per alert: broadcast via hub, email recipients via `EmailNotifier`, optional image upload via `AlertImageStorageService`.

> **Finding a method**: each mixin file's docstring lists its methods. `grep -rn "def <method>" application/services/notification/` jumps straight to the owning file.

#### [overlay_normalize.py](application/services/overlay_normalize.py) — shared overlay helpers

Consolidates 6 helpers + `_coerce_int` that previously existed in 4 different files (`domain/model_pipeline.py`, `application/services/notification.py`, `application/services/clip_storage.py`, `routes/clips_routes.py`). Imported by all 4 of those files plus the new [notification/overlay_helpers.py](application/services/notification/overlay_helpers.py).

Exports:
- `_coerce_int(value)` — `int(value)` with `TypeError`/`ValueError` → `None`.
- `_normalize_track_id(value)` — alias of `_coerce_int`.
- `_normalize_overlay_box(raw)` — accepts dict, list/tuple, **or** an object with `x1`/`y1`/`x2`/`y2` attrs. Returns `{"x1","y1","x2","y2"}` or `None`.
- `_normalize_box_norm(raw)` — accepts dict or object with `x`/`y`/`w`/`h`. Returns `{"x","y","w","h"}` or `None`.
- `_normalize_overlay_detection(raw_detection)` — produces the canonical dict shape with `cls_name`, `conf`, `box`, optional `box_norm`, optional `track_id`.
- `_overlay_detection_base_key(normalized)` / `_append_overlay_detection(...)` — used to merge frames across camera + clip windows while de-duplicating by `(class, conf, x1, y1, x2, y2)` and reconciling tracked-vs-untracked detections.

The version here is the **superset** of all four pre-split implementations (accepts dict + list/tuple + object inputs). The dict-only call sites are unaffected; the more permissive helper means future code can pass Pydantic detection objects directly.

#### [tracker.py](application/services/tracker.py)
- `Track` dataclass — id, class, bbox, score, timestamps, hit/miss counters, `confirmed` flag, constant-velocity `predict()`, `update()`.
- `ByteTrackLite` — two-stage matching (high-conf first, low-conf flicker pass), Hungarian via SciPy when available, greedy fallback.
- `ROIAlertEngine` — normalised polygon stored per camera; `check()` runs bbox ∩ polygon (cross-product test) → emits `AlertRaisedEvent`.

#### [edgeinference.py](application/services/edgeinference.py) — `EdgeInferenceClient`
HTTP client for the Jetson TensorRT API.

- Configurable endpoint paths (`EDGE_ADD_PATH`, `EDGE_PATCH_PATH`, `EDGE_DELETE_PATH`, `EDGE_API_KEY`).
- Methods: `add_camera`, `patch_camera`, `delete_camera`, `health`, `list_cameras`, `reconcile`.
- Custom error: `EdgeCameraInventoryError`.

#### [webrtcgateway.py](application/services/webrtcgateway.py) — `WebRTCGatewayClient`
Two modes:
1. **Admin API enabled** — POST/DELETE on MediaMTX `/v3/config/paths/add` to actually provision paths.
2. **Disabled** — derive stable public WHEP URLs without admin calls.

Methods: `provision_stream`, `delete_stream`, `resolve_public_url`. Env: `WEBRTC_PUBLIC_BASE_URL`, `WEBRTC_ADMIN_API_URL`, `WEBRTC_ADMIN_API_ENABLED`, `MTX_API_USER/PASS`, `WEBRTC_ADMIN_TIMEOUT_S`.

#### [stream_gateway.py](application/services/stream_gateway.py)
Helpers `gateway_path()` and `build_playback()` to produce playback JSON `{type, mode, path, url}` for the frontend. Env: `STREAM_GATEWAY_PUBLIC_BASE`, `STREAM_GATEWAY_PLAYBACK_MODE`, `STREAM_GATEWAY_PATH_PREFIX`.

#### [alert_image_storage.py](application/services/alert_image_storage.py)
Azure Blob Storage for alert JPEGs. `store_alert_image()` decodes a data URI, uploads to blob, returns a 24-h SAS URL. `delete_alert_image()` by storage key.

#### [clip_storage.py](application/services/clip_storage.py)
Azure Blob Storage for `.mp4` clips. Mirror of the image service — `request_clip`, `upload_clip`, `delete_clip`.

#### [retention.py](application/services/retention.py) — `RetentionService`
`purge(clip_retention_days, alert_retention_days)` — deletes old `VideoRecord` rows + blobs, then old `Notification` rows + alert blobs. Returns `{clips_deleted, alerts_deleted}`. Invoked by the background loop in [main.py](main.py).

#### [user_snapshot_cache.py](application/services/user_snapshot_cache.py)
LRU + TTL cache for `(id, user_name, email)`. Used by the SSE route to avoid hitting the DB on every new subscriber.

#### [pipeline_cache.py](application/services/pipeline_cache.py)
In-memory cache of loaded `ModelPipeline` objects keyed by user_id, used by `Manager`.

### 5.3 Channels — `application/channels/`

#### [channel_config.py](application/channels/channel_config.py) — `VideoChannelConfig`
Unified runtime config for a camera. Extends the base `ChannelConfig` ABC defined in the same file (previously lived in the root-level `dto.py`; folded back in during the refactor). Fields cover IDs, stream URLs, metadata, enabled flags, schedule (day_of_week, start/end_time, list of windows), detection tuning (sample_fps, decode_backend, resize, emit_format, jpeg_quality, reconnect/poll timeouts, detection_path_template), and the tri-state inheritance flags. `normalize_schedule()` parses a flexible schedule list.

#### [channel.py](application/channels/channel.py) — `VideoChannel`
Polls the Jetson for the latest detection JSON per camera.
- `key()` — string of camera_uuid.
- `detection_urls()` / `detection_stream_urls()` — candidate URLs with last-success fallback.
- `fetch_detection_json()` — GET `detection/{camera_uuid}/latest`, parses detections/tracks/alerts, retries with backoff.
- `_device_reachable` flag — drives exponential backoff in `Manager`.

### 5.4 Models — `application/models/`

#### [vision_config.py](application/models/vision_config.py)
`VisionTask` enum (`object_detection`, `pose_estimation`, `multi_task`) and `VisionModelConfig` Pydantic model — the vendor-agnostic shape used by [domain/model.py](domain/model.py)'s `VisionModel` Protocol. Previously lived in the root-level `dto.py`.

#### [yolo_config.py](application/models/yolo_config.py)
`YoloModelConfig` (extends `VisionModelConfig`) — task, device, imgsz, conf, iou, allowed_labels, plus YOLO-specific `det_weights`, `pose_weights`, `half`, `allowed_det_labels`, `skeleton_label`, `remote_url`, `keypoints_format`.

#### [yolo_model.py](application/models/yolo_model.py)
`YoloMultiTaskModel` implements the `VisionModel` Protocol. `infer()` calls the Jetson `/v1/detect|/v1/pose|/v1/multitask` endpoints and returns `DetectionsProducedEvent` or an inference-failure event. Throttles pose at a configurable interval when running multi-task.

### 5.5 Builder — `application/builder/`

#### [pipeline_builder.py](application/builder/pipeline_builder.py)
`PipelineBuilder.create(template)` — instantiates a `VideoChannel` per channel config, optionally loads `YoloMultiTaskModel`, returns a `ModelPipeline`.

---

## 6. Routes Layer (HTTP API)

All routes are mounted under `/api`. Auth is enforced via `Depends(get_current_user)` unless noted.

### 6.0 Shared route helpers

#### [routes/_background.py](routes/_background.py)

Two small helpers used across the camera/site/user delete paths (previously duplicated 3× each):

- `_spawn_bg_task(coro, *, name)` — `asyncio.create_task` with a done-callback that logs `SUCCESS` / `CANCELLED` / `FAILED with error: ...` under the helper's own logger.
- `_delete_blobs_background(keys, *, service_cls, label)` — parallel blob deletion in batches of 10 with progress logs (`Starting deletion of N L blobs`, per-batch counts, `COMPLETE: deleted X failed Y`).

`extract_notification_clip_storage_keys(payload)` lives in [application/services/clip_storage.py](application/services/clip_storage.py) (previously duplicated in `routes/user_routes.py`, `routes/camera_routes.py`, `routes/site_routes.py`, `application/services/retention.py`). It walks a notification payload looking for `storage_key` values in `msg`, `msg.clip`, `extra`, `extra.clip`, `extra.multi_camera_prerecordings[]`, and `clip` — used at delete-time to also clean up the associated clip blobs.

### 6.1 Auth — [routes/auth/](routes/auth/)

[routes/auth/__init__.py](routes/auth/__init__.py) aggregates the auth routers.

#### [signup.py](routes/auth/signup.py)

| Method | Path | Body | Returns | Notes |
|---|---|---|---|---|
| POST | `/auth/signup/request` | `SignupRequestCode {user_name, user_email}` | `{signup_token}` | Generates OTP, hashes it (`code_hash`), stores `EmailVerification` + `SignupTempData`, sends email |
| POST | `/auth/signup/verify` | `SignupVerifyRequest {signup_token, code}` | `AuthTokenOut {access_token, user}` | Validates OTP (TTL + attempts), creates User, returns JWT |
| POST | `/auth/login` | `LoginRequest {user_email, password}` | `AuthTokenOut` | bcrypt verify, updates last_login |
| POST | `/auth/verify-email` | — | — | Resend OTP |

Env: `OTP_TTL_SECONDS`, `OTP_MAX_ATTEMPTS`, `SIGNUP_TEMP_TTL_SECONDS`, `AUTH_DEBUG_RETURN_OTP` (returns OTP in response — dev only).

#### [oauth.py](routes/auth/oauth.py)
Placeholders for third-party OAuth providers.

### 6.2 Cameras — [camera_routes.py](routes/camera_routes.py)

| Method | Path | Purpose |
|---|---|---|
| GET | `/cameras` | List all cameras for the user (across sites) |
| GET | `/cameras/{camera_uuid}` | Fetch single camera (with config + devices) |
| POST | `/cameras` | Create via `Manager.add_camera()` — validates site/device ownership, auto-links sole device if only one on site |
| PATCH | `/cameras/{camera_uuid}` | `Manager.edit_camera()` — patch semantics, structural changes force pipeline reset |
| DELETE | `/cameras/{camera_uuid}` | `_cleanup_camera_runtime()` — pipeline removal → edge delete → WebRTC unprovision → DB cascade |
| GET | `/cameras/{camera_uuid}/snapshot` | Fetches JPEG from Jetson with URL fallbacks; short-lived cache |
| GET | `/cameras/{camera_uuid}/stream` | **SSE** — polls Jetson and emits `event: detection\ndata: {…}` per frame |

### 6.3 Clips — [clips_routes.py](routes/clips_routes.py)

| Method | Path | Purpose |
|---|---|---|
| GET | `/clips` | Paginated/filterable VideoRecord list scoped to user's cameras |
| GET | `/clips/{clip_id}` | Single clip (ownership-checked) |
| DELETE | `/clips/{clip_id}` | Mark deleted, enqueue blob cleanup task |
| DELETE | `/clips` | Bulk delete by ID list |

### 6.4 Sites — [site_routes.py](routes/site_routes.py)

| Method | Path | Purpose |
|---|---|---|
| GET | `/sites` | Lists Sites (`is_deleted=False`) for user |
| GET | `/sites/{site_uuid}` | Site + cameras + devices |
| POST | `/sites` | Create; auto-generates `site_code`; initialises `SiteSettings` |
| PATCH | `/sites/{site_uuid}` | Update name/address/timezone |
| DELETE | `/sites/{site_uuid}` | Soft-delete; background task cascades notification/clip cleanup |
| GET | `/sites/{site_uuid}/settings` | Schedule + prerecord config |
| PUT | `/sites/{site_uuid}/settings` | Update schedule + prerecord config |
| GET | `/sites/{site_uuid}/devices` | Linked devices |
| POST | `/sites/{site_uuid}/devices/{device_uuid}` | Attach device (M:N) |
| DELETE | `/sites/{site_uuid}/devices/{device_uuid}` | Detach |

### 6.5 Devices — [device_routes.py](routes/device_routes.py)

| Method | Path | Purpose |
|---|---|---|
| GET | `/devices` | List user's devices |
| GET | `/devices/{device_uuid}` | Single device |
| POST | `/devices` | Create; auto-generates `device_code` |
| PATCH | `/devices/{device_uuid}` | Update url/name/enabled |
| DELETE | `/devices/{device_uuid}` | `Manager.cleanup_device_resources()` — tears down all cameras on device |
| GET | `/devices/{device_uuid}/health` | Proxies `EdgeInferenceClient.health()` |
| POST | `/devices/{device_uuid}/reconcile` | Sync DB vs edge inventory; `?dry_run=true` for preview |

### 6.6 Notifications — [notifications_routes.py](routes/notifications_routes.py)

| Method | Path | Purpose |
|---|---|---|
| GET | `/notifications` | Paginated, filterable by site/camera/visible/unread |
| GET | `/notifications/sse` | **SSE** — Bearer token via header or `?token=`; subscribes to `WebNotificationHub` |
| POST | `/notifications/{id}/read` | Set `read_at` |
| POST | `/notifications/{id}/unread` | Clear `read_at` |
| DELETE | `/notifications/{id}` | Soft-delete (`visible=False`) |
| DELETE | `/notifications` | Bulk soft-delete |

### 6.7 Users — [user_routes.py](routes/user_routes.py)

| Method | Path | Purpose |
|---|---|---|
| GET | `/users/me` | `UserOut {id, user_name, user_email}` |
| PUT | `/users/me` | Update profile; invalidate user-snapshot cache |
| POST | `/users/me/password` | Change password (verifies current) |
| DELETE | `/users/me` | Soft-delete user; background cascade |

### 6.8 Notification Emails — [notification_email_routes.py](routes/notification_email_routes.py)

| Method | Path | Purpose |
|---|---|---|
| GET | `/notification-emails/sites/{site_uuid}` | List recipients |
| POST | `/notification-emails` | Add recipient |
| DELETE | `/notification-emails/{id}` | Remove recipient |

---

## 7. Scripts, Deployment & Tests

### Scripts
- [scripts/add_provided_device.py](scripts/add_provided_device.py) — onboards a pre-configured Jetson into the DB.
- [scripts/migrate_notifications_schema.py](scripts/migrate_notifications_schema.py) — historical bulk migration with retry.

### Deployment
- [Dockerfile](Dockerfile) — image build for the API.
- [setup_mysql.sh](setup_mysql.sh) — local MySQL bootstrap.
- [deployment/deploy.sh](deployment/deploy.sh) — Azure deploy script (build → push ACR → trigger).
- [deployment/config.prod.sh](deployment/config.prod.sh) — prod env var template.
- [.github/workflows/deploy-prod.yml](.github/workflows/deploy-prod.yml) — CI/CD.

### Tests
- [tests/test_camera_delete_cleanup.py](tests/test_camera_delete_cleanup.py) — camera delete tears down pipeline + edge + WebRTC in correct order.
- [tests/test_notification_persistence_commit.py](tests/test_notification_persistence_commit.py) — notifications flushed + committed atomically, rolled back on failure.
- [tests/test_model_pipeline_stream_notifications.py](tests/test_model_pipeline_stream_notifications.py) — full detection-to-notification lifecycle.
- [tests/test_site_delete_cleanup.py](tests/test_site_delete_cleanup.py) — site soft-delete triggers cascading cleanup.
- [tests/test_edge_client_timeouts.py](tests/test_edge_client_timeouts.py) — `EdgeInferenceClient` handles timeouts gracefully.
- [tests/test_detection_failure_passthrough.py](tests/test_detection_failure_passthrough.py) — `InferenceFailedEvent` propagates correctly.
- [tests/test_tensort_broadcaster_filtering.py](tests/test_tensort_broadcaster_filtering.py), [tests/test_tensort_pipeline_autosizing.py](tests/test_tensort_pipeline_autosizing.py) — TensorRT pipeline (documented separately in [Tensort_documetn.md](Tensort_documetn.md)).

---

## 8. End-to-End Flows

### Flow 1 — Signup + Verification + Login

1. **`POST /api/auth/signup/request`** — `user_repository.exists_email()` check, generate OTP (random 6-digit), `code_hash = sha256(code)`, insert `EmailVerification(email, code_hash, expires_at)` + `SignupTempData`, send SMTP email, return `{signup_token}`.
2. **`POST /api/auth/signup/verify`** — Validate `signup_token`, fetch `EmailVerification`, compare `sha256(code)` vs stored `code_hash`, check `expires_at` + attempt count. On success: `user_repository.create_user(email_verified=True)`, delete OTP rows, `create_access_token({user_id})`, return `{access_token, user}`.
3. **`POST /api/auth/login`** — fetch user, `verify_password()`, `update_last_login_at`, return `{access_token, user}`.

### Flow 2 — Add Camera + Start Pipeline + Notify Edge

1. **`POST /api/cameras`** with `CameraCreateSchema`.
2. Route validates site ownership + device ownership (auto-link if site has exactly one device, otherwise `device_uuid` required).
3. `Manager.add_camera()`:
   - `channel_repository.upsert_camera_from_channel_config()` creates Camera + ChannelConfiguration + PipelineCamera in a single transaction.
   - Instantiates `VideoChannelConfig` and a `VideoChannel`.
   - Registers the channel with the in-memory `ModelPipeline` (or spawns one).
   - `EdgeInferenceClient.add_camera()` POSTs config to Jetson.
   - `WebRTCGatewayClient.provision_stream()` configures MediaMTX path.
   - Emits a `ChannelCreateEvent`.
4. Returns `CameraWithConfigSchema`.

### Flow 3 — Detection → Notification → Email → Image Storage

1. `ModelPipeline` polls Jetson detection endpoint every `poll_interval_ms`. `VideoChannel.fetch_detection_json()` returns `{detections, pose, tracks, alerts}`.
2. `NotificationService.on_detections_produced()`:
   - `ByteTrackLite.update()` — assign detections → tracks; promote to `confirmed` after N hits.
   - `ROIAlertEngine.check()` — bbox ∩ ROI test → `AlertRaisedEvent` if intersecting.
3. For each alert / confirmed track (per trigger mode):
   - `notification_repository.get_camera_context()` denormalises user/site/camera/device + tri-state modes (single query).
   - `AlertImageStorageService.store_alert_image()` uploads JPEG → SAS URL embedded in `payload.extra`.
   - `notification_repository.insert_notification()` flushes a row; service commits with rollback on failure.
4. Fan-out:
   - `WebNotificationHub.broadcast(msg)` — pushes to all SSE subscribers for that user.
   - `EmailNotifier.send_email()` — async fire-and-forget to site recipients.
5. Background `RetentionService.purge()` deletes rows + blobs after TTL.

### Flow 4 — Clip Recording + Retrieval

1. Site setting `config.prerecord_cameras` enables prerecording for a camera.
2. `ModelPipeline` maintains an in-memory last-N-seconds frame buffer.
3. On a detection + trigger mode match (`any_detection` or `roi_enter`), a clip job is enqueued.
4. `ClipStorageService.upload_clip()` writes the encoded MP4 to Azure Blob, returns a SAS URL.
5. `VideoRecord` row created with `status='completed'`, `storage_key`, `recording_url`, `overlay_payload`.
6. `GET /api/clips` queries `VideoRecord` rows for the user's cameras.
7. `DELETE /api/clips/{id}` marks the row and enqueues blob deletion via `app.state.alert_blob_cleanup_tasks`.

### Flow 5 — WebRTC Live Playback

1. On camera create, `WebRTCGatewayClient.provision_stream()` configures a MediaMTX path (admin API mode) or simply resolves the public WHEP URL (no-admin mode).
2. Camera record carries `webrtc_url` (or it is derived from `camera_code`).
3. `GET /api/cameras/{camera_uuid}` returns the URL. Frontend POSTs an SDP offer to the WHEP endpoint; MediaMTX handles RTSP-ingest → WebRTC-egress.

### Flow 6 — Edge Reconciliation Loop

1. `_edge_reconcile_loop` ticks every `EDGE_RECONCILE_INTERVAL_S`.
2. `Manager.reconcile_all_devices_edge()` iterates enabled devices and, for each, calls `EdgeInferenceClient.list_cameras()`.
3. Diff vs DB:
   - Cameras in DB but missing on edge → re-add via `add_camera`.
   - Cameras on edge but not in DB → if `EDGE_RECONCILE_DELETE_UNKNOWN`, delete them.
4. Failures track an `_edge_failure_streak` per device with exponential backoff (base 5 s, factor 1.5, capped at 300 s).

---

## 9. Cross-Cutting Concerns

### Authentication & Authorisation
- All protected routes use `Depends(get_current_user)` which decodes the Bearer JWT (`HS256`, 24 h TTL) and loads the `User` ORM.
- Ownership checks (camera/site/device belongs to user) happen at the route layer before delegating to services.

### Async Session Lifecycle
- `db_manager.AsyncSessionLocal` is an `async_sessionmaker`.
- `expire_on_commit=False` — ORM objects remain usable after commit (relied on heavily by the verify-flow logic).
- Routes use `Depends(get_async_db)` for per-request session and explicitly call `await db.commit()` / `await db.rollback()`.
- Services that span multiple sessions (Manager, background loops) inject the `session_factory` and open scopes themselves.
- **Bulk `update()` must include `.execution_options(synchronize_session=False)`** — see CLAUDE.md for the historical incident.

### Error Handling
- Route layer raises `HTTPException` with appropriate status codes (400, 401, 403, 404, 409, 503).
- Service layer raises typed errors (`EdgeDeviceUnavailableError`, `EdgeCameraInventoryError`, `ValueError`) which routes translate.
- Background loops catch + log all exceptions to avoid crashing the task.

### Background Tasks
| Task | Cadence | Cancellable | Purpose |
|---|---|---|---|
| `_edge_reconcile_loop` | `EDGE_RECONCILE_INTERVAL_S` (30 s default) | Yes (lifespan shutdown) | Sync DB ↔ edge cameras |
| `_retention_cleanup_loop` | `RETENTION_CLEANUP_INTERVAL_S` (3600 s) | Yes | Purge old clips/notifications |
| `app.state.alert_blob_cleanup_tasks` | On-demand | Awaited at shutdown | Async blob deletion after route returns |

### Caching
- **`UserSnapshotCache`** — LRU + TTL for user metadata used in SSE streams.
- **`pipeline_cache`** — in-memory loaded `ModelPipeline` per user_id.
- **`VideoChannel`** — caches last-successful detection URL to skip the fallback chain.
- **Azure SAS URLs** — 24 h expiry; the frontend may cache them.

### Real-Time Delivery
- `WebNotificationHub` is an async pub-sub keyed by user_id.
- The SSE route at `/api/notifications/sse` subscribes for the lifetime of the HTTP connection and emits `event: notification\ndata: {…}\n\n` messages.
- Detection-streaming uses a similar SSE shape at `/api/cameras/{id}/stream`.

### Tri-State Inheritance
`Camera.notification_trigger_mode` and `Camera.camera_playback_enabled` use `"inherit" | <specific> | <specific>`. When `"inherit"`, the effective value is resolved from the site-level settings. This pattern is implemented in `notification_repository._coerce_trigger_mode()` and consumed throughout the notification service.

### Soft Deletes
- `Site.is_deleted`, `User.is_deleted` (where applicable), `Notification.visible`.
- Background cascades batch the physical cleanup (blob deletions, dependent rows) to keep DELETE endpoints responsive.

---

## 10. Refactoring Notes & History

### 10.1 What was done (refactor pass — 2026-05-11)

Six passes against the working tree, each verifiable independently. No public API or external import path changed.

**Pass 1 — Dead code + DTO split**
- Archived `Backend/Updated/` (5 abandoned drafts predating current code) to `/home/peshal/1886NOENTRY/_archive/Updated_2026-05-11/`.
- Deleted: `Backend/utils.py` (no importer), `Backend/verify_fix.py` (one-off mock test, not in `tests/`), `Backend/core/utils.py` (empty file).
- Deleted `Backend/dto.py`. Its contents moved to their respective folders:
  - `VisionTask`, `VisionModelConfig` → [application/models/vision_config.py](application/models/vision_config.py)
  - `ChannelConfig` (abstract) → folded into [application/channels/channel_config.py](application/channels/channel_config.py)
  - Orphan `SignupRequestCode` / `SignupConfirm` / `LoginRequest` (signup.py had its own real versions) → removed
  - Duplicate `utc_now()` (canonical lives in [core/database_orm.py](core/database_orm.py)) → removed
- 4 import sites updated; 1 dead import removed from [pipeline_repository.py](application/repositories/pipeline_repository.py).

**Pass 2 — `utc_now()` dedup**
- [user_repository.py](application/repositories/user_repository.py) defined its own `utc_now()`. Replaced with `from core.database_orm import utc_now` (matching every other repository).

**Pass 3a — Shared env-var helpers**
- New [core/env.py](core/env.py) with `env_bool` / `env_int` / `env_float`. Replaces duplicate `_env_*` parsers in 7 files (`main.py`, `routes/auth/signup.py`, `core/database.py`, `application/services/edgeinference.py`, the new `application/services/notification/` package, `application/repositories/verify_repository.py`, `application/channels/channel.py`).
- One behavior preservation: `edgeinference.py`'s old `_env_float` defaulted `minimum=0.1`; the shared version takes `minimum: Optional` (default `None`). Every edgeinference call site now passes `minimum=0.1` explicitly.

**Pass 3b — Shared overlay-normalize helpers**
- New [application/services/overlay_normalize.py](application/services/overlay_normalize.py) consolidates 6 helpers (`_normalize_track_id`, `_normalize_overlay_box`, `_normalize_box_norm`, `_normalize_overlay_detection`, `_overlay_detection_base_key`, `_append_overlay_detection`) + `_coerce_int`. Previously duplicated **24 times** (each of 6 helpers in each of 4 files: `domain/model_pipeline.py`, `notification.py`, `clip_storage.py`, `routes/clips_routes.py`).
- The unified version is the **superset** of all four pre-split variants (accepts dict, list/tuple, and object inputs). Dict-only callers continue to work; new code can pass Pydantic detection objects directly.

**Pass 3c — `extract_notification_clip_storage_keys` consolidation**
- Moved to [application/services/clip_storage.py](application/services/clip_storage.py) as a public function. Removed duplicate copies from `routes/user_routes.py`, `routes/camera_routes.py`, `routes/site_routes.py`, `application/services/retention.py`. All 4 sites now import it under the existing private alias `_extract_notification_clip_storage_keys` to keep call sites unchanged.

**Pass 3d — Shared route background helpers**
- New [routes/_background.py](routes/_background.py) with `_spawn_bg_task` + `_delete_blobs_background`. Previously each duplicated 3× across `routes/{user,site,camera}_routes.py` with subtly different log verbosity. Unified version uses the most informative logging (the `user_routes.py` blob cleanup now also emits informational "Starting / Batch N / COMPLETE" log lines — no functional change).

**Pass 4 — `notification.py` split (2821 lines → 11-file package, max 564 lines/file)**

`notification.py` became `notification/` ([details in Section 5.2](#notification--notificationservice--webnotificationhub--emailnotifier-package)). External imports unchanged. `NotificationService` composed via inheritance from 5 mixin classes that share `self._*` state set up by `__init__` in `service.py`.

**Pass 5 — `manager.py` split (2074 lines → 10-file package, max 498 lines/file)**

Same pattern as notification: `manager.py` → `manager/` package, `Manager` composed of 6 mixins ([details in Section 5.2](#manager--manager-package)).

### 10.2 Mixin pattern — what to know

Each service package follows the same shape:
- `service.py` — single entry point: imports the mixins, declares `class X(Mixin1, Mixin2, ...): __init__: ...`.
- `_service_<concern>.py` — each mixin file owns one cohesive concern. Methods read/write `self._*` state set up by `service.py`'s `__init__`. A mixin class is not standalone — it's only valid as a base of the composed class.
- `types.py`, `helpers.py`, etc. — module-level stuff (Pydantic models, dataclasses, constants, free functions) lives outside the mixin classes.
- `__init__.py` — re-exports the public surface so external `from application.services.<pkg> import X` calls keep working unchanged.

**To find a method**: each mixin file's module docstring names the methods it owns. Or just `grep -rn "def <method>" application/services/<package>/`.

### 10.3 Remaining refactor candidates (not yet done)

1. **Configuration** — replace ad-hoc `os.environ` reads in [core/config.py](core/config.py) and across services with a single `pydantic_settings.BaseSettings` object. The env-var parsing in [core/env.py](core/env.py) is a stepping stone but not the same thing.
2. **Migrations** — the hand-written migration block in [core/database.py](core/database.py) is large. Introducing Alembic would make schema changes auditable and reversible.
3. **Sync/async duality** — `DatabaseManager` still exposes a sync engine. Most paths are async; the few remaining sync usages could be migrated and the sync engine removed.
4. **Repository transactions** — the "caller controls commit" rule is consistent but undocumented; a `UnitOfWork` wrapper would make this explicit and reduce route-layer commit boilerplate.
5. **OTP storage** — the historical `code` column being nullable while `code_hash` is the real source of truth is migration debt that could be cleaned up.
6. **Test coverage** — there is meaningful integration test coverage of the destructive paths (delete/cascade) but light coverage of create/edit happy paths and zero coverage of routes themselves. A FastAPI `TestClient` smoke suite per router would catch regressions cheaply.
7. **Mixin import bloat** — each `_service_*.py` mixin file imports a generous superset of types/clients to be self-contained. `ruff --fix` (or `autoflake`) will trim the unused imports cleanly when desired.
8. **Folder rename to strict DDD** — the current `core/` / `routes/` layout maps cleanly onto `infrastructure/` / `interfaces/` but the rename touches every import statement. Deferred as cosmetic.

---

*Generated as a refactoring + onboarding aid. File references use line numbers as of the current state of the repo and may drift after edits.*
