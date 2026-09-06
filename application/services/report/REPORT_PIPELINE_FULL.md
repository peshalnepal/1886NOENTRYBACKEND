# The Complete Report Pipeline (Backend) — End to End

This is the master document. It follows a report from the **moment an AI camera
detects something** all the way to a **member viewing the PDF in the browser**,
naming every file, function, and DB table on the path.

Read this top-to-bottom once and the two companion docs fill in the fine detail:
- `REPORT_SERVICE_DETAILED.md` — every function of `PdfReportGenerator` + the repositories, with examples.
- `REPORT_GENERATOR.md` — the `pdf_builder.py` PDF-drawing internals.

---

## The big picture in one diagram

```
  ┌─────────────────────────────────────────────────────────────────────────┐
  │ STAGE A — an ALERT is born (upstream, not the report code)               │
  │   camera → AI pipeline → NotificationService.enqueue → flusher batch     │
  │   → INSERT into `notifications` table (approval_status="pending")        │
  └─────────────────────────────────────────────────────────────────────────┘
                                   │  a pending alert now sits in the DB
                                   ▼
  ┌─────────────────────────────────────────────────────────────────────────┐
  │ STAGE B — an OPERATOR REVIEWS it                                          │
  │   POST /api/notifications/{id}/approve   (routes/notifications_routes.py) │
  │   → _decide_notifications → notif_repo.set_approval(approved_by=<op>)     │
  │   → row becomes approval_status="approved", visible=1                     │
  │   → operator may add NOTES (payload["notes"]) + choose "approve+email"    │
  └─────────────────────────────────────────────────────────────────────────┘
                                   │  approved alerts are the INPUT to reports
                                   ▼
  ┌─────────────────────────────────────────────────────────────────────────┐
  │ STAGE C — a REPORT is GENERATED  (this module)                           │
  │   3 possible triggers, all land in PdfReportGenerator.build_report():    │
  │     1. human clicks "Send"  → POST /api/reports/approved-alerts          │
  │     2. daily scheduler       → ReportScheduler (every 60s)               │
  │     3. operator "approve+email" → _queue_urgent_report (fire-and-forget) │
  │                                                                          │
  │   build_report:  read org+sites+cameras → read approved `notifications`  │
  │     → flatten each to _AlertEntry → download snapshot images             │
  │     → mint report_uuid → draw PDF (pdf_builder) → ReportResult           │
  └─────────────────────────────────────────────────────────────────────────┘
                          │                         │
              persist=True│                         │email transport set
                          ▼                         ▼
  ┌───────────────────────────────────┐  ┌──────────────────────────────────┐
  │ STAGE D — ARCHIVE                 │  │ STAGE E — EMAIL                  │
  │  _persist_report                 │  │  email.send_report(pdf,to=[...]) │
  │  → ReportRepository.create       │  │  (SMTP; org members + site        │
  │  → INSERT `organization_reports` │  │   notification recipients)        │
  │    (report_uuid + pdf_data blob) │  └──────────────────────────────────┘
  └───────────────────────────────────┘
                          │
                          ▼
  ┌─────────────────────────────────────────────────────────────────────────┐
  │ STAGE F — a MEMBER BROWSES/VIEWS it later                                │
  │   GET /api/reports            → ReportRepository.list_reports (no blob)  │
  │   GET /api/reports/{id}/view  → ReportRepository.get_with_pdf → inline   │
  │   GET /api/reports/{id}/download → same, as attachment                   │
  └─────────────────────────────────────────────────────────────────────────┘
```

Everything below is the same six stages, expanded.

---

## STAGE A — where the alert (the raw material) comes from

The report is a roll-up of **`notifications`** rows. Those rows are created *by a
different subsystem* — you don't need to master it, just know the shape it
produces:

- A camera's AI pipeline detects an object and calls
  `NotificationService.enqueue_notification(msg, ctx, extra_payload)`
  (`application/services/notification/service.py`).
- The **flusher** (`application/services/notification/flusher.py`) batches these
  and INSERTs `Notification` rows via `NotificationCreateDTO`, storing everything
  the report later reads inside the JSON `payload` column as
  `{"msg": {...}, "extra": {...}}` — title, `cls_names`, `max_conf`,
  `image_url`, `clip_url`, camera name, etc.
- A fresh alert is created **`approval_status="pending"`** and **`visible=0`** —
  it is *not* in a report yet. It's waiting for an operator.

**Table `notifications` — the columns the report reads:** `id`, `site_uuid`,
`camera_uuid`, `detected_at`, `title`, `message`, `event_type`, `approved_by`,
`approved_at`, `visible`, `approval_status`, and the JSON `payload`.

---

## STAGE B — the operator turns "pending" into "approved"

File: `routes/notifications_routes.py`. This is the gate that decides what a
report is allowed to contain.

- **`POST /api/notifications/{id}/approve`** (and the bulk `POST /api/notifications/approve`)
  - Gated by `RequirePermission(Permission.ALERTS_APPROVE)` (operators + admins).
  - Calls the shared helper **`_decide_notifications(db, ctx, ids, approve=True, email_owner=?)`**:
    1. `notif_repo.set_approval(db, ids=..., approval_status="approved", visible=True, approved_by=<operator id>, site_uuids=<org scope>)` — flips the rows. **`approved_by` is now set** — this is the flag the report later checks for `operator_approved_only`.
    2. `db.commit()`.
    3. Publishes the now-visible alert to the end-user's realtime stream.
    4. If the operator chose **"approve + email"** (`email_owner=True`):
       - `notification_service.queue_approved_emails(...)` — emails the site recipients immediately, **and**
       - `_queue_urgent_report(ctx, ids, session_factory)` — fires an **urgent report** in the background (Stage C, trigger 3).
- **Operator notes**: during review the operator can attach structured notes, stored back into `payload["notes"]` as a list of `{text, action, attributes{...}, author_name, created_at}`. These become the report's **Notes** and **Action Taken** rows.

After Stage B, an approved alert row looks like §1a in `REPORT_SERVICE_DETAILED.md`.

**Two flavors of approval → two report types:**
| Operator action | Effect on the alert | Which report it lands in |
|---|---|---|
| Approve (plain) | approved, visible, **not** emailed | the daily **general** report |
| Approve **+ email** | approved, visible, **emailed now** | an **urgent** report, archived immediately |

---

## STAGE C — generating the report

All three triggers converge on **`PdfReportGenerator.build_report()`**. The
class lives in `application/services/report/pdf_report_service.py`; its every
function is documented (with examples) in `REPORT_SERVICE_DETAILED.md`. Here we
focus on **who triggers it and how it's wired**.

### The three triggers

**Trigger 1 — a human clicks "Send report"**
`routes/report_routes.py` → `POST /api/reports/approved-alerts`
- Gated by `RequirePermission(Permission.REPORTS_SEND)` (admins + operators).
- `_build_generator(session_factory, notification_service)` constructs a `PdfReportGenerator`, pulling the **email transport** and dashboard URL off the app's `NotificationService`.
- Calls `generate_and_send(org_id, report_type, window, persist=True, generated_by=<caller>)` → build **+ archive + email**.
- Sibling `POST /api/reports/approved-alerts/download` calls `build_report(...)` only (no email, no archive) and streams the PDF back for a live preview.

**Trigger 2 — the daily scheduler**
`application/services/report/report_scheduler.py` → `ReportScheduler`, started in `main.py`'s `lifespan`:
```python
def _report_generator_factory() -> PdfReportGenerator:
    return PdfReportGenerator(session_factory=SessionLocal, email=email_notifier,
                              dashboard_base_url=DASHBOARD_URL)
report_scheduler = ReportScheduler(session_factory=SessionLocal,
                                   generator_factory=_report_generator_factory,
                                   poll_s=env_float("REPORTS_SCHEDULER_POLL_S", 60))
if env_bool("REPORTS_SCHEDULER_ENABLED", True):
    report_scheduler.start()
```
- Every `poll_s` seconds, `_tick()` reads all enabled schedules (`ReportScheduleRepository.list_enabled`), and for each org checks: has today's `send_hour:send_minute` passed in the org's timezone, and is `last_sent_on != today`?
- If due: `_send_for_org()` first **stamps `last_sent_on` = today** (`ReportScheduleRepository.mark_sent`, committed) so a slow/failed SMTP can't double-send, then calls `generate_and_send(report_type="general", window_hours=<sched>, persist=True)`.
- The schedule itself (send time, timezone, window, on/off) is configured by admins via `GET/PUT /api/reports/schedule` → `ReportScheduleRepository.upsert`.

**Trigger 3 — operator "approve + email" (the urgent report)**
`routes/notifications_routes.py` → `_queue_urgent_report(...)` (shown in Stage B):
- Builds a `PdfReportGenerator` with **`email=None`** (the alert email already went out — the report is archive-only), and `asyncio.create_task(generator.build_report(report_type="urgent", notification_ids=<the approved ids>, persist=True))`.
- Fire-and-forget: approval never blocks on PDF work; the task is tracked in a module-level `set` so it isn't garbage-collected.

### What `build_report()` does (recap — full detail in the companion doc)

1. **One DB session**, read up front: org (`OrganizationRepository.get_by_id`), owner name for "Prepared for" (`_resolve_owner_name`), all sites (`list_org_sites`) → name/timezone/address maps, camera locations (inline SQL), the **approved alert rows** (`NotificationRepository.list_notifications` with `approval_status="approved"` + window/type filters), and the recipient email list (`_resolve_recipients`). Session closes.
2. **Flatten** each `Notification` → `_AlertEntry` (`_entry_from_row`), skipping non-operator-approved rows when `operator_approved_only`.
3. **Download snapshots** concurrently (`_attach_images`).
4. **Mint one `report_uuid`** — printed on the PDF *and* stored in the DB row (that's why the printed ID is searchable).
5. **Draw the PDF** (`_render_pdf` → `pdf_builder.PDFReport`).
6. Return a **`ReportResult`** (pdf bytes + counts + recipients + uuid + filename).

### The notification query that selects report content

`NotificationRepository.list_notifications` (in `notification_repository.py`) is
the actual SQL. The report calls it two ways:
- **urgent by ids:** `ids=<notification_ids>, site_uuids=<org sites>, approval_status="approved"`.
- **general/urgent by window:** `site_uuids=<org sites>, approval_status="approved", only_visible=True, emailed=(type=="urgent"), detected_after=start, detected_before=end`.

So a report can only ever contain **approved, visible** alerts belonging to the
**org's own sites**, inside the window — the Stage-B gate is enforced again here.

---

## STAGE D — archiving the PDF

Only when `persist=True`. `_persist_report` (in the service) builds a
`ReportCreateDTO` and calls **`ReportRepository.create(db, dto)` then `db.commit()`**.

- File: `application/repositories/report_repository.py`.
- Table: **`organization_reports`**. Key columns: `id`, **`report_uuid`** (CHAR(36), unique, indexed — the shareable id), `org_id`, `report_type`, `filename`, `generated_by`/`generated_by_email`, `site_uuids` (JSON list), `period_start`/`period_end`, `alert_count`, **`pdf_data`** (LONGBLOB — the whole PDF), `pdf_size`, `created_at`.
- Because the blob is multi-MB, list queries never select it (`_LIST_COLUMNS`); only `get_with_pdf` loads it.

---

## STAGE E — emailing the PDF

Only in `generate_and_send`, and only if an `email` transport was injected and
`recipients` is non-empty.

- Recipients = org members' emails + enabled per-site notification emails + any extras, de-duped (`_resolve_recipients`).
- The service builds subject/HTML/text (`_subject`, `_email_html`, `_email_text`) and calls the transport:
  `email.send_report(subject, html_body, text_body, attachment_bytes=pdf, attachment_filename=..., to_emails=recipients)`
  (`application/services/notification/email_notifier.py`), which sends over SMTP.
- Result: `result.emailed = True/False`. Failure is logged, never raised.

(Note: the **urgent** background report from Stage B uses `email=None`, so it
archives but does not re-email — the alert email already went out.)

---

## STAGE F — a member browses & views the archive later

File: `routes/report_routes.py`, all gated by `RequirePermission(Permission.ORG_READ)`
(any member), org-scoped.

- **`GET /api/reports`** → `ReportRepository.list_reports(...)` → a list of blob-free `ReportOut` rows. Filters: `report_type`, `start`/`end` (created_at), `site_uuid`, `operator_email`, and **`report_uuid`** (search by the printed ID). This feeds the frontend's spreadsheet table.
- **`GET /api/reports/{id}/view`** → `_load_report_pdf` → `ReportRepository.get_with_pdf` → returns the PDF with `Content-Disposition: inline` → the frontend shows it in an iframe (read without downloading).
- **`GET /api/reports/{id}/download`** → same load, `Content-Disposition: attachment` → browser download.

---

## Who calls what — the full backend call map

```
main.py lifespan
├── builds NotificationService (email_notifier lives here)
└── builds ReportScheduler(generator_factory=_report_generator_factory)  ── Trigger 2

routes/notifications_routes.py
└── approve → _decide_notifications
              ├── NotificationRepository.set_approval           (Stage B)
              └── _queue_urgent_report → PdfReportGenerator.build_report  ── Trigger 3

routes/report_routes.py
├── POST /approved-alerts          → PdfReportGenerator.generate_and_send ── Trigger 1
├── POST /approved-alerts/download → PdfReportGenerator.build_report
├── GET  /                          → ReportRepository.list_reports        (Stage F)
├── GET  /{id}/view                 → ReportRepository.get_with_pdf         (Stage F)
├── GET  /{id}/download             → ReportRepository.get_with_pdf         (Stage F)
└── GET/PUT /schedule               → ReportScheduleRepository.get/upsert

ReportScheduler._tick (every 60s)
├── ReportScheduleRepository.list_enabled
└── _send_for_org
    ├── ReportScheduleRepository.mark_sent
    └── PdfReportGenerator.generate_and_send

PdfReportGenerator.generate_and_send
└── build_report
    ├── OrganizationRepository.get_by_id / list_org_sites / list_org_members
    ├── NotificationRepository.list_notifications / list_notification_email_rows
    ├── _entry_from_row · _attach_images · _render_pdf(→ pdf_builder.PDFReport)
    └── _persist_report → ReportRepository.create → db.commit               (Stage D)
    └── (back in generate_and_send) email.send_report                       (Stage E)
```

---

## The tables involved

| Table | Written by | Read by the report pipeline | Holds |
|---|---|---|---|
| `notifications` | flusher (Stage A), approval (Stage B) | `build_report` (the content) | one alert; `payload` JSON carries notes/media/classes |
| `organization_reports` | `ReportRepository.create` (Stage D) | list/view/download (Stage F) | the archived PDF blob + `report_uuid` metadata |
| `organization_report_schedules` | `ReportScheduleRepository.upsert` / `mark_sent` | `ReportScheduler` (Stage C, trigger 2) | per-org daily send time + `last_sent_on` |

---

## Permissions & scoping recap

- **`ALERTS_APPROVE`** (`alerts:approve`) — approve/reject alerts (Stage B). Operators + admins.
- **`REPORTS_SEND`** (`reports:send`) — generate/send reports (Stage C, trigger 1). Operators + admins.
- **`ORG_READ`** (`org:read`) — browse/view/download the archive (Stage F). Any member.
- Everything is **org-scoped**: a normal caller acts on their own org (`OrgContext`); a platform admin (no single org) must pass `org_id`. The notification query is always constrained to the org's own `site_uuids`.

---

## Config knobs (env)

| Var | Default | Stage | Effect |
|---|---|---|---|
| `REPORTS_SCHEDULER_ENABLED` | true | C-2 | Turn the daily scheduler on/off. |
| `REPORTS_SCHEDULER_POLL_S` | 60 | C-2 | How often the scheduler checks for due orgs. |
| `REPORT_MAX_ALERTS` | 500 | C | Max alert rows per report. |
| `REPORT_MAX_IMAGES` | 300 | C | Max snapshots embedded (0 = none). |
| `REPORT_IMAGE_FETCH_CONCURRENCY` | 8 | C | Parallel snapshot downloads. |
| `REPORT_IMAGE_FETCH_TIMEOUT_S` | 15 | C | Per-download timeout. |

---

## One concrete trace (general daily report)

1. **08:00 America/New_York** — `ReportScheduler._tick` sees org #5's schedule is enabled, 08:00 has passed, `last_sent_on` is yesterday → due.
2. `_send_for_org` stamps `last_sent_on="2026-07-09"` and commits (so it can't fire twice today).
3. `generate_and_send(org_id=5, report_type="general", window_hours=24, persist=True)`.
4. `build_report` opens a session: org #5, its 3 sites, cameras; queries `notifications` where site ∈ those 3, `approval_status="approved"`, `visible=1`, `emailed=0`, `detected_at` in the last 24h → 12 rows; resolves 4 recipient emails.
5. Flattens 12 → `_AlertEntry`; downloads 9 of 12 snapshots; mints `report_uuid=3f2c1e9a…`; renders the PDF.
6. `_persist_report` → `ReportRepository.create` → `organization_reports` row id 88 (with the blob + uuid).
7. Back in `generate_and_send`: `email.send_report(pdf, to=[4 emails])` → `emailed=True`.
8. Next morning a member opens the Reports table, searches `3f2c1e9a`, clicks **View** → `GET /api/reports/88/view` → `get_with_pdf` → inline PDF in an iframe.
```
```
