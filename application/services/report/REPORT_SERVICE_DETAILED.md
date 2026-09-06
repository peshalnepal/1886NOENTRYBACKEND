# Report Service — Deep Walkthrough (with examples)

This document explains, in plain language and with concrete examples, four files:

1. `application/services/report/pdf_report_service.py` — **the brain** (`PdfReportGenerator`)
2. `application/repositories/report_repository.py` — **saves/reads the finished PDF** (`ReportRepository`)
3. `application/repositories/report_schedule_repository.py` — **stores the "email me daily at 8am" setting** (`ReportScheduleRepository`)
4. `application/services/report_creater.py` — **a dead, unused stub** (`AlertReport`)

> Terminology note: there is no separate "Notification service" in these files.
> What you are looking at is the **PdfReportGenerator**. It *uses* the
> `NotificationRepository` (to read alert rows) and it is *wired* with an email
> transport that comes from the app's `NotificationService`. But the report
> logic itself all lives in `PdfReportGenerator`. Wherever this doc says
> "the service", it means `PdfReportGenerator`.

---

## 0. The 10,000-foot view

A **report** is a PDF that lists every alert an operator approved for one
organization, over some time window. The service:

```
takes:  org_id + time window + report_type
does:   read the org's sites → read approved alert rows → pull out
        (notes, image, clip) from each alert → download the snapshot images →
        draw a PDF → (save it to the DB) → (email it)
gives:  a ReportResult (pdf bytes + counts + the recipients + a UUID)
```

Two entry points call it:

| Caller | Method used | Emails? | Saves to DB? |
|---|---|---|---|
| `POST /api/reports/approved-alerts` (a human clicks "Send report") | `generate_and_send()` | yes | yes |
| `ReportScheduler` (runs every 60s, fires the daily digest) | `generate_and_send()` | yes | yes |
| `POST /api/reports/approved-alerts/download` (a human clicks "Preview") | `build_report()` | no | no |

---

## 1. The data objects (what flows through the pipeline)

Before the functions, understand the 3 shapes the data takes.

### 1a. Input: a `Notification` DB row (an alert)

The service reads alert rows. The important part is the JSON `payload` column.
A single alert row looks roughly like this:

```python
Notification(
    id=4021,
    site_uuid=UUID("aaaa...."),
    camera_uuid=UUID("cccc...."),
    detected_at=datetime(2026, 7, 9, 14, 3, 0, tzinfo=utc),
    approved_by=17,                 # operator user id (None = auto-approved)
    approved_at=datetime(2026, 7, 9, 14, 5, 0, tzinfo=utc),
    title="Person in restricted zone",
    message="Motion detected at loading dock",
    event_type="intrusion",
    payload={
        "msg": {
            "title": "Person in restricted zone",
            "body": "Motion detected at loading dock",
            "camera_name": "Dock Cam 2",
            "cls_names": ["person"],
            "max_conf": 0.91,
            "image_url": "https://cdn.example.com/snap/4021.jpg",
            "clip_url":  "https://cdn.example.com/clip/4021.mp4",
        },
        "extra": {},
        "notes": [                  # what the operator typed during review
            {
                "text": "Confirmed trespasser, called security",
                "action": "Dispatched guard to dock",
                "attributes": {"object_type": "person",
                               "gender": "male",
                               "clothing_color": "red hoodie",
                               "direction": "heading north"},
                "author_name": "Jane (operator)",
                "created_at": "2026-07-09T14:04:30Z"
            }
        ],
    },
)
```

### 1b. Intermediate: `_AlertEntry` (a flattened, clean alert)

The service converts each messy `Notification` row into a tidy `_AlertEntry`
dataclass — a private struct used only for rendering. For the row above:

```python
_AlertEntry(
    detected_at=datetime(2026,7,9,14,3,tzinfo=utc),
    title="Person in restricted zone",
    body="Motion detected at loading dock",
    site_uuid="aaaa....",
    site_name="Warehouse A",           # looked up from the sites map
    site_location="12 Dock Rd",
    site_tz="America/New_York",
    camera_name="Dock Cam 2",
    camera_location="Loading dock",
    alert_type="intrusion",
    classes=["person"],
    max_conf=0.91,
    notes=[{...the note dict above...}],
    image_url="https://cdn.example.com/snap/4021.jpg",
    clip_url="https://cdn.example.com/clip/4021.mp4",
    approved_at=datetime(2026,7,9,14,5,tzinfo=utc),
    image_jpeg=None,                    # filled in later by _attach_images
)
```

### 1c. Output: `ReportResult` (the return value)

```python
ReportResult(
    report_type="general",
    report_id=88,                       # DB row id (None if not saved)
    report_uuid="3f2c1e9a-...-9b21",    # printed on the PDF AND stored in DB
    org_id=5,
    org_name="Acme Security",
    alert_count=12,
    image_count=9,                      # how many alerts had a usable snapshot
    site_count=3,
    start=datetime(...), end=datetime(...),
    recipients=["a@x.com", "b@y.com"],
    emailed=True,
    pdf_bytes=b"%PDF-1.5....",          # the actual file
    filename="noentry-general-report-2026-07-09-1405-3f2c1e9a.pdf",
)
```

---

## 2. `PdfReportGenerator` — the service, function by function

### Call tree (who calls whom)

```
generate_and_send()                      ← public entry (build + email)
└── build_report()                       ← public entry (build only)
    ├── OrganizationRepository.get_by_id()
    ├── _resolve_owner_name()            → "Prepared for" name
    ├── OrganizationRepository.list_org_sites()
    ├── (inline SQL) Camera locations
    ├── NotificationRepository.list_notifications()   → the alert rows
    ├── _resolve_recipients()            → email list
    │   ├── OrganizationRepository.list_org_members()
    │   └── NotificationRepository.list_notification_email_rows()
    ├── _entry_from_row()   (per alert)  → _AlertEntry
    │   ├── _extract_image_url()
    │   └── _extract_clip_url()
    ├── _attach_images()                 → downloads snapshots
    │   └── _fetch_image_bytes()   → normalize_to_jpeg() (from pdf_builder)
    ├── _render_pdf()                    → pdf bytes
    │   ├── _render_record()  (per alert)
    │   │   ├── _detection_text()
    │   │   ├── _notes_text()   → _bullet(), _primary_class(), _classify()
    │   │   ├── _actions_text()
    │   │   ├── _fmt_dt_site()
    │   │   └── (PDFReport draw calls: field_row/image_block/…)
    │   ├── _fmt_window()
    │   └── _fmt_dt()
    ├── _filename()
    └── _persist_report()                → ReportRepository.create() + commit
        └── builds ReportCreateDTO
back in generate_and_send():
    ├── _subject(), _email_text(), _email_html()
    └── email.send_report(...)
```

---

### `__init__(self, *, session_factory, email=None, dashboard_base_url=None)`

Sets the service up. It does **not** do any work yet.

- **Inputs**
  - `session_factory` — a function that opens a DB session: `async with session_factory() as db:`.
  - `email` — an object with a `send_report(...)` coroutine, or `None` to disable email.
  - `dashboard_base_url` — optional base URL (stored, currently informational).
- **Also does:** creates the three repositories it will use (`NotificationRepository`, `OrganizationRepository`, `ReportRepository`) and reads 4 env tunables (`REPORT_MAX_ALERTS=500`, `REPORT_MAX_IMAGES=300`, `REPORT_IMAGE_FETCH_CONCURRENCY=8`, `REPORT_IMAGE_FETCH_TIMEOUT_S=15`).
- **Output:** the constructed object.

---

### `generate_and_send(...) → ReportResult` — public, "do everything"

The top-level entry used by the route and the scheduler.

- **Inputs (all keyword-only):**
  | name | type | meaning | example |
  |---|---|---|---|
  | `org_id` | int | which org | `5` |
  | `report_type` | str | `"general"` or `"urgent"` | `"general"` |
  | `start`, `end` | datetime? | time window | `None, None` |
  | `window_hours` | int? | shortcut: `start = now - hours` | `24` |
  | `operator_approved_only` | bool | drop auto-approved alerts | `True` |
  | `extra_recipients` | list[str]? | extra emails | `["boss@acme.com"]` |
  | `notification_ids` | list[int]? | urgent: exactly these alerts | `None` |
  | `persist` | bool | save to DB | `True` |
  | `generated_by` / `generated_by_email` | int?/str? | who triggered it | `17 / "jane@acme.com"` |

- **Steps:**
  1. If `window_hours` given and `start` is `None`: `start = now - window_hours` (uses `_timedelta_hours`).
  2. `result = await self.build_report(...)` — does all the building.
  3. If `email is None` → log and return (no send).
  4. If `result.recipients` empty → log and return (nobody to send to).
  5. Build subject/bodies (`_subject`, `_email_html`, `_email_text`) and call `email.send_report(...)`.
  6. Set `result.emailed = True/False`; on exception, log and set `False`.
- **Output:** the `ReportResult` (now with `emailed` set).
- **Example call:**
  ```python
  result = await gen.generate_and_send(
      org_id=5, report_type="general", window_hours=24,
      persist=True, generated_by=17, generated_by_email="jane@acme.com")
  # result.emailed == True, result.report_id == 88
  ```

---

### `build_report(...) → ReportResult` — public, "build only, no email"

**This is the real pipeline.** Everything else is a helper. Same inputs as
above minus `window_hours` (and `persist` defaults to `False`).

Walk through it in order:

1. **Normalize type:** `report_type` becomes exactly `"urgent"` or `"general"`.
2. **Open ONE session** (`async with self._session_factory() as db:`) and inside it:
   - `org = OrganizationRepository.get_by_id(db, org_id)` → the org row.
   - `org_name` = `org.name` or `"Organization 5"`.
   - `prepared_for = _resolve_owner_name(...)` → the owner's display name (for the PDF "Prepared for").
   - `sites = OrganizationRepository.list_org_sites(db, org_id)` → all sites in the org.
   - Build 3 lookup dicts keyed by `site_uuid`: **name**, **location/address**, **timezone**.
   - `site_uuids` = list of those keys.
   - Run one inline SQL `SELECT camera_uuid, location FROM cameras WHERE site_uuid IN (...)` → **camera location** dict.
   - **Select the alert rows** (`NotificationRepository.list_notifications`), two modes:
     - **urgent + `notification_ids`:** exactly those ids, scoped to the org's sites, `approval_status="approved"`.
     - **otherwise:** `approval_status="approved"`, `only_visible=True`, `emailed=(report_type=="urgent")`, plus the `detected_after=start`/`detected_before=end` window.
   - `recipients = _resolve_recipients(...)` → the email list.
   - *(session closes here)*
3. **Convert rows → entries** (outside the session): for each row, skip it if `operator_approved_only` and `row.approved_by is None`; else `_entry_from_row(...)` → `_AlertEntry`. Collect into `entries`.
4. **Derive window for urgent-by-ids:** if `notification_ids` was used and no `start/end` was given, set `start, end = min/max(detected_at)` across the entries.
5. **Download images:** `await self._attach_images(entries)` fills `entry.image_jpeg`. Then `image_count = number of entries that got an image`.
6. **Mint the UUID:** `report_uuid = str(uuid.uuid4())`. **This same string is printed on the PDF AND saved to the DB row**, which is why you can search a report by the ID printed on it.
7. **Render:** `pdf_bytes = self._render_pdf(report_uuid=..., entries=..., ...)`.
8. **Assemble `ReportResult`** (with `report_id=None`, `filename=_filename(...)`).
9. **If `persist`:** `result.report_id = await self._persist_report(...)`.
10. **Return** `result`.

- **Output:** `ReportResult` (with `pdf_bytes` populated; `report_id` set only if persisted).
- **Why the session closes before rendering:** image downloading and PDF drawing are slow/CPU work; holding a DB connection open across them would waste a pooled connection. All DB reads are done up front.

---

### `_persist_report(*, result, entries, generated_by, generated_by_email) → int | None`

Saves the finished PDF to the archive.

- **Inputs:** the finished `result`, the `entries` (to derive which sites appear), and who generated it.
- **Steps:**
  1. Build a `ReportCreateDTO` (org_id, report_uuid, type, filename, generated_by[_email], `site_uuids` = the distinct sites in the entries, period_start/end, alert_count, and `pdf_data=result.pdf_bytes`).
  2. Open a session, `ReportRepository.create(db, dto=dto)`, `db.commit()`.
  3. Return the new row's `id`.
- **On any exception:** log and return `None` (a failed archive must not break the email path).
- **Output:** the new report row id, or `None`.

---

### Data-extraction helpers

#### `_entry_from_row(row, site_name_by_uuid, site_location_by_uuid, site_tz_by_uuid=None, camera_location_by_uuid=None) → _AlertEntry`
Flattens one `Notification` row into an `_AlertEntry` (see §1b).
- Reads `row.payload`, then its `msg` and `extra` sub-dicts (defensively — non-dicts become `{}`).
- `notes` ← `payload["notes"]` (list, else `[]`).
- `classes` ← `msg["cls_names"]`; `max_conf` ← `float(msg["max_conf"])` (or `None`).
- Looks up `site_name/location/tz` and `camera_location` from the maps passed in.
- Uses `_extract_image_url` / `_extract_clip_url` for the media links.
- **Output:** one `_AlertEntry`.

#### `_resolve_owner_name(db, *, org, org_name) → str`
Returns the org owner's `user_name` for the "Prepared for" line; falls back to `org_name` if there's no owner or the lookup fails.

#### `_extract_image_url(msg, extra) → str | None`
First non-empty of `msg["image_url"]`, then `extra["image_url"]`.

#### `_extract_clip_url(msg, extra) → str | None`
`msg["clip_url"]`, else `extra["clip"]["recording_url"]`.

#### `_resolve_recipients(db, *, org_id, site_uuids, extra) → list[str]`
Builds the email audience: **org members' emails** + **enabled per-site
notification emails** + **extras**, then de-dupes case-insensitively preserving
first-seen order.
- Example: members `["jane@acme.com","BOB@acme.com"]` + site emails
  `["bob@acme.com","guard@site.com"]` + extra `["boss@acme.com"]`
  → `["jane@acme.com","BOB@acme.com","guard@site.com","boss@acme.com"]`
  (the second `bob` is dropped as a case-insensitive duplicate).

---

### Image fetching

#### `_attach_images(entries) → None` (mutates entries)
Downloads snapshots concurrently and stores the JPEG on each entry.
- Skips entirely if `REPORT_MAX_IMAGES <= 0`.
- Picks the first `max_images` entries that have an `image_url`.
- Uses an `asyncio.Semaphore(concurrency)` + one shared `httpx.AsyncClient` so at most N downloads run at once.
- For each: `raw = _fetch_image_bytes(...)` → `normalize_to_jpeg(raw)` → store in `entry.image_jpeg`.
- Errors are swallowed (`return_exceptions=True`) — a missing image never fails the report.

#### `_fetch_image_bytes(client, url) → bytes | None`
Turns a URL into raw bytes.
- `data:` URLs — decodes base64 or percent-encoding.
- `http(s)` URLs — GETs with the shared client; returns bytes on HTTP 200.
- Anything else / any error → `None`.

---

### PDF rendering (turns entries into a `PDFReport`)

#### `_render_pdf(*, report_uuid, report_type, prepared_for, entries, start, end, site_count) → bytes`
Lays out the whole document:
1. `doc.brand_header(...)` (logo + wordmark + timestamp), `doc.title_center(TYPE)`.
2. Info block via `field_row`: **Report ID** (the uuid), **Prepared for**, **Reporting window** (`_fmt_window`), **Sites covered**, **Approved alerts**. *(No organization name — by request.)*
3. If no entries: an "OBSERVATIONS — 0 RECORDS" band + a "nothing found" sentence, then `return doc.render()`.
4. Else: an "OBSERVATIONS — N RECORDS" band, then `_render_record(doc, i, entry)` for each entry.
5. `return doc.render()` → the raw PDF `bytes`.

#### `_render_record(doc, idx, entry) → None`
Draws one alert block: a numbered grey band (`#3 — Person in restricted zone`),
then `field_row`s for Time (site-local via `_fmt_dt_site`), Site, Site Address,
Camera, Type (`_detection_text`), Alert Message, Notes (`_notes_text`), Action
Taken (`_actions_text`), Reviewed; then the photo (`image_block` if downloaded,
else a link) and a Video link.

#### `_detection_text(entry) → str`
Example output: `"Intrusion · Detected: person · 91% confidence"`.

#### `_notes_text(entry) → str`  (the class-aware one)
Turns the operator's structured note into bullet lines. It picks a field set
based on `attributes.object_type` (or guesses via `_classify(entry.classes)`):
- **vehicle** → Type / Model / Color / Direction
- **person** → Type / Gender / Clothing / Direction
- **other** → Color / Direction

For the example note in §1a it produces:
```
• Type: Person
• Gender: male
• Clothing: red hoodie
• Direction: heading north
• Note: Confirmed trespasser, called security
• By: Jane (operator)
```
If there are no notes at all, it falls back to `• Detected: person`.

#### `_actions_text(entry) → str`
Collects each note's `action` field as bullets, e.g. `"• Dispatched guard to dock"`.

#### `_classify(classes) → "vehicle" | "person" | "other"` (module fn)
Buckets detected classes; `["car"] → "vehicle"`, `["person"] → "person"`, else `"other"`.

#### `_bullet(lines, label, value)` / `_primary_class(entry)`
Tiny helpers: append `• Label: value` only when value is non-empty; titlecase the first class name.

---

### Email + formatting helpers

- `_subject(result) → str` — e.g. `"[1886NOENTRY] General Alert Report — Acme Security (12 alerts)"` (adds `[URGENT]` for urgent).
- `_email_text(result) → str` / `_email_html(result) → str` — plain and HTML email bodies (these **still include the org name**; only the PDF dropped it).
- `_fmt_dt(dt) → str` — `"2026-07-09 14:05:00"` in UTC (`"n/a"` if not a datetime).
- `_fmt_dt_site(dt, tz_name) → str` — event time in the site's timezone, e.g. `"2026-07-09, 10:03 AM (EDT)"`.
- `_fmt_window(start, end) → str` — `"All approved alerts"` or `"… to … UTC"`.
- `_filename(report_type, report_uuid) → str` — `"noentry-general-report-2026-07-09-1405-3f2c1e9a.pdf"`.
- `_timedelta_hours(hours)` (module fn) — `timedelta(hours=…)`.

---

## 3. `ReportRepository` — saving & reading the finished PDF

Persistence for the `organization_reports` table. **Stateless. Never commits**
(the caller wraps the transaction). The PDF blob can be several MB, so list
queries deliberately avoid selecting it.

`_LIST_COLUMNS` = every column **except `pdf_data`** (id, report_uuid, org_id,
type, filename, generated_by[_email], site_uuids, period_start/end, alert_count,
pdf_size, created_at).

### `create(db, *, dto: ReportCreateDTO) → OrganizationReport`
- **Input:** a session + a `ReportCreateDTO` (org_id, report_uuid, report_type, filename, generated_by[_email], site_uuids, period_start/end, alert_count, pdf_data).
- **Does:** builds an `OrganizationReport` ORM object (computing `pdf_size = len(pdf_data)`), `db.add(row)`, `await db.flush()` (assigns the `id` but does **not** commit).
- **Output:** the new `OrganizationReport` (with `.id` populated after flush).
- **Example:**
  ```python
  async with session_factory() as db:
      row = await ReportRepository().create(db, dto=dto)
      await db.commit()          # caller commits
      new_id = row.id            # e.g. 88
  ```

### `list_reports(db, *, org_id, report_type=None, created_after=None, created_before=None, generated_by_email=None, site_uuid=None, report_uuid=None, limit=100, offset=0) → list[Row]`
- **Input:** a session + filters. All filters are optional except `org_id`.
- **Does:** `SELECT <list columns> WHERE org_id = :org` and conditionally adds `report_uuid`, `report_type`, `created_at >= after`, `created_at < before`, `generated_by_email`. Orders newest-first (`created_at desc, id desc`). `offset`/`limit` (limit clamped to 1–500). Then, if `site_uuid` was given, filters the returned page **in Python** — keeps rows whose `site_uuids` JSON list contains that site (report volume is tiny, so no DB-specific `JSON_CONTAINS`).
- **Output:** a list of blob-free rows (each has the `_LIST_COLUMNS` attributes; **no `pdf_data`**).
- **Example — "find the report whose printed ID is X":**
  ```python
  rows = await ReportRepository().list_reports(
      db, org_id=5, report_uuid="3f2c1e9a-...-9b21")
  # rows[0].filename, rows[0].alert_count, rows[0].pdf_size, ...
  ```

### `get_with_pdf(db, *, report_id, org_id=None) → OrganizationReport | None`
- **Input:** a session + the numeric `report_id` (+ optional `org_id` to enforce ownership).
- **Does:** `SELECT * FROM organization_reports WHERE id = :id [AND org_id = :org]` and returns the **full** row **including `pdf_data`**.
- **Output:** the ORM row or `None`.
- **Used by:** the `/download` and `/view` endpoints — this is the *only* method that loads the multi-MB blob.

---

## 4. `ReportScheduleRepository` — the "email me daily" setting

Persistence for `organization_report_schedules` (one row per org). Also
stateless / no commit (except the background scheduler, which commits around it).
A schedule row holds: `org_id`, `is_enabled`, `send_hour`, `send_minute`,
`timezone`, `window_hours`, `operator_approved_only`, `last_sent_on`.

### `get(db, *, org_id) → OrganizationReportSchedule | None`
- **Input:** session + org id. **Output:** that org's schedule row, or `None` if it never set one.
- `SELECT ... WHERE org_id = :org LIMIT 1`.

### `list_enabled(db) → list[OrganizationReportSchedule]`
- **Input:** session. **Output:** every schedule with `is_enabled = TRUE`.
- **Used by:** `ReportScheduler._tick()` every minute to find which orgs might be due.

### `upsert(db, *, org_id, is_enabled=None, send_hour=None, send_minute=None, timezone=None, window_hours=None, operator_approved_only=None, last_sent_on=None) → OrganizationReportSchedule`
- **Input:** session + org id + any subset of fields. **Only the fields you pass (non-`None`) are changed** — this is a partial update.
- **Does:** `get()` the row; if none, create a new `OrganizationReportSchedule(org_id=...)` and `db.add` it. Then set each provided field. `await db.flush()` (no commit).
- **Output:** the created/updated row.
- **Example — "enable daily 8:00am America/New_York digest":**
  ```python
  async with session_factory() as db:
      row = await ReportScheduleRepository().upsert(
          db, org_id=5, is_enabled=True, send_hour=8, send_minute=0,
          timezone="America/New_York", window_hours=24)
      await db.commit()      # the route commits
  ```

### `mark_sent(db, *, org_id, date_str) → None`
- **Input:** session + org id + a date string like `"2026-07-09"` (in the schedule's own timezone).
- **Does:** sets `last_sent_on = date_str` and flushes. This is the **idempotency stamp** — the scheduler writes it so each day fires exactly once even though it polls every minute.
- **Output:** none.

---

## 5. `report_creater.py` — a DEAD, UNUSED stub ⚠️

```python
class AlertReport():
    def __init__(self, *args, **kwds):
        site_repo = SiteRepository()            # created then discarded (local var)
        notification_repo = NotificationRepository()
    def create_report(self, site_uuid):
        notificaitions_by_time = None           # does nothing
    def publish_report(self, report): pass
    def save_report(self, report): pass
    def __call__(self, *args, **kwds): pass
```

- **What it does:** nothing. Every method is empty or a no-op; the repos it makes in `__init__` are local variables that are immediately thrown away (not even stored on `self`).
- **Is it used?** No — nothing imports `AlertReport`. The real generator is `PdfReportGenerator` in `pdf_report_service.py`.
- **Recommendation:** it's an abandoned early scaffold. Safe to delete (or ignore). Note the typo `notificaitions_by_time` too. **Do not confuse this file with the working service.**

---

## 6. End-to-end example: a manual "Send report" click

1. Frontend → `POST /api/reports/approved-alerts` with `{report_type:"general", window_hours:24}`.
2. Route builds a `PdfReportGenerator` (wiring in the SMTP transport from the app's `NotificationService`) and calls `generate_and_send(org_id=5, report_type="general", window_hours=24, persist=True, generated_by=17, generated_by_email="jane@acme.com")`.
3. `generate_and_send` sets `start = now-24h`, calls `build_report`.
4. `build_report` reads org #5, its 3 sites, its cameras; reads the approved+visible+not-emailed alerts detected in the last 24h (say 12 rows); resolves 4 recipient emails; flattens the 12 rows to `_AlertEntry`s; downloads 9 of the 12 snapshots; mints `report_uuid=3f2c1e9a-…`; renders the PDF; saves it via `_persist_report` (→ DB id 88).
5. Back in `generate_and_send`: builds the email subject/body and calls `email.send_report(...)` to all 4 recipients; sets `result.emailed=True`.
6. Route returns `{report_id:88, alert_count:12, image_count:9, site_count:3, recipient_count:4, emailed:true, filename:"noentry-general-report-…-3f2c1e9a.pdf"}`.
7. Later, a member opens the Reports table (`GET /api/reports`), searches the UUID `3f2c1e9a`, and clicks **View** → `GET /api/reports/88/view` → `get_with_pdf` loads the blob → shown inline in an iframe.
