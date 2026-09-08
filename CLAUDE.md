# Backend — CLAUDE.md

## Role
Act as a senior backend developer: expert in Python, FastAPI, and asyncio, with
deep MySQL and SQLAlchemy (2.x async) knowledge and a strong grasp of
domain-driven, layered architecture. Prefer correct async/transaction handling
and respect the layer boundaries below.

## Stack
- FastAPI + SQLAlchemy 2.x **async** (aiomysql) + MySQL
- Auth: JWT bearer (HS256), bcrypt passwords; RBAC via a permission catalog
- Pydantic v2 for HTTP schemas, DTOs, and pipeline events

## Architecture (layered / DDD-influenced)
Request flow: **HTTP → route → DTO → repository / service → ORM → MySQL**.

- **`routes/`** — HTTP boundary (FastAPI routers, mounted under `/api` in
  `main.py`). Validate the request with a `core/schemas.py` model, gate with
  `RequirePermission`, translate to a DTO, then call a repository/service.
  Routes never touch the ORM directly through raw SQL logic.
- **`core/`** — cross-cutting foundation:
  - `schemas.py` — Pydantic request/response models (the HTTP contract; single
    source of truth, imported by routes).
  - `database_orm.py` — SQLAlchemy ORM models (incl. RBAC `Permission`/role tables).
  - `database.py` — `DatabaseManager`: async/sync session factories + table
    init. `create_all` never alters existing tables — schema changes to existing
    tables need manual migration code here (`migrations/` holds helpers).
  - `config.py`, `env.py`, `security/` (hashing, JWT tokens, roles + permission catalog).
- **`domain/`** — vendor-agnostic domain layer: `events.py` (Pydantic pipeline
  events: `RTSPEvent`, `DetectionsProducedEvent`, …) and `channel.py`.
- **`application/`** — application layer:
  - `dtos.py` — internal DTOs (repo inputs / service-to-service), deliberately
    separate from HTTP schemas. Convention: `*CreateDTO` / `*UpdateDTO` /
    `*UpsertDTO`; patch with `model_dump(exclude_unset=True)`.
  - `repositories/` — persistence, one per aggregate (Site, Channel/Camera,
    Device, Notification, Organization, User, Pipeline, Video, Verify). Stateless
    (instantiated per call, e.g. `SiteRepository()`), take an `AsyncSession`, and
    **do not commit** — the caller owns the transaction. Exception: streaming
    bulk-delete helpers take a session *factory* and manage their own batches.
  - `services/` — orchestration & infrastructure: `manager/` (pipeline + edge-
    device orchestration, app singleton), `notification/` (hub / flusher / email
    / clip), `authz_service.py`, plus pipeline, tracker, edge inference, clip &
    image storage, retention, webrtc/stream gateways, caches.
  - `channels/` — `VideoChannelConfig` + runtime. There is no local inference
    layer: detection runs on the Jetson edge (`tensort/`) and the backend
    consumes its detection streams.
- **`interface/`** — currently empty placeholder.

## Auth & RBAC
- JWT bearer → `get_current_user` (dependencies.py) loads the `User` via `UserRepository`.
- `OrgContext` resolves the caller's organization + role.
- `RequirePermission(Permission.X)` is the **single** authz dependency: it gates
  the request and returns the `OrgContext`. Flat routes (`/sites`, `/cameras`,
  `/devices`) resolve the caller's single org; path routes (`/orgs/{org_id}/…`,
  `/sites/{site_uuid}/…`) check the path context. Platform admins and effective
  org admins short-circuit the check.

## Conventions
- Use the async session from the `get_async_db` dependency; sessions use
  `expire_on_commit=False`.
- Validate at the HTTP boundary (`core/schemas.py`), convert to a DTO, then call
  a repository — keep ORM and raw SQL out of routes.
- Pydantic v2 only (`ConfigDict`, `model_validator`, `field_validator`,
  `model_dump`/`model_validate`). Avoid the deprecated v1 `class Config`.
- App entry: `main.py` (`lifespan` initializes the DB, builds the `Manager`,
  `NotificationService`, and `RetentionService` singletons, and starts the
  background pipeline / edge-reconcile / retention loops).
