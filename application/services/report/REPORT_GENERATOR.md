# PDF Report Generator — Developer Documentation

The **report subsystem** rolls up an organization's operator-approved alerts —
with operator notes, alert snapshot images, and clip links — into a single
self-contained PDF, archives it in MySQL, and emails it to the org's members and
per-site recipients. It has **no third-party PDF dependency**: the PDF is
hand-written to the PDF 1.5 spec.

---

## 1. Architecture & data flow

```
                         HTTP boundary                Application layer                    Persistence
                         ─────────────                ─────────────────                    ───────────
POST /api/reports/…  →  report_routes.py  →  PdfReportGenerator (pdf_report_service.py)
                                                │
                                                ├─ OrganizationRepository  ─┐
                                                ├─ NotificationRepository  ─┤→  AsyncSession → MySQL
                                                ├─ ReportRepository       ─┘   (organization_reports)
                                                │
                                                ├─ _attach_images()  → httpx → snapshot bytes → JPEG
                                                │
                                                ├─ PDFReport (pdf_builder.py)   → raw PDF bytes
                                                │
                                                └─ email.send_report()          → SMTP

ReportScheduler (report_scheduler.py)  → (every 60 s) → PdfReportGenerator.generate_and_send()
```

**Layer roles** (per `Backend/CLAUDE.md`):
- `report_routes.py` — HTTP boundary: validates request, gates with `RequirePermission`, resolves org, delegates.
- `pdf_report_service.py` — application service: gathers data, orchestrates, renders, archives, emails. Stateless apart from its session factory.
- `pdf_builder.py` — infrastructure: a minimal PDF writer (text flow, JPEG embedding, links, vector logo).
- `report_repository.py` — persistence for the `organization_reports` archive. Stateless; never commits.
- `report_scheduler.py` — background loop that fires the daily "general" report per org.

---

## 2. File-by-file reference

### 2.1 `pdf_builder.py` — the dependency-free PDF writer

A top-down flowing document model. `self.y` is the cursor measured **down from
the top margin**; helpers append content and advance it, auto-paging when space
runs out. A4 page (`595.28 × 841.89` pt), `MARGIN = 50`.

#### Module-level functions

| Function | Input | Output | Purpose |
|---|---|---|---|
| `_char_width(ch, size, bold)` | char, font size (pt), bold flag | `float` (pt) | Advance width of one glyph from the baked Helvetica AFM tables (`_HELV`/`_HELV_BOLD`). Fallback `556` for non-ASCII. |
| `text_width(s, size, bold=False)` | string, size, bold | `float` (pt) | Sum of glyph widths — drives word-wrap and link-rect sizing. |
| `_escape_pdf_text(s)` | `str` | `bytes` | Escapes `\ ( )`, maps common typographic chars to WinAnsi code points, replaces anything else with `?`. Keeps the byte stream in sync. |
| `_jpeg_info(data)` | raw image `bytes` | `(w, h, components)` or `None` | Parses JPEG SOF markers to read geometry without decoding pixels. |
| `normalize_to_jpeg(raw)` | arbitrary image `bytes` | `(jpeg_bytes, w, h, comps)` or `None` | Passes real JPEGs straight through (they go into the PDF's `DCTDecode` filter untouched); otherwise transcodes via Pillow **if installed**, else returns `None`. |

#### `PDFReport` class

Cursor & page management:

| Method | Input | Output | Purpose |
|---|---|---|---|
| `__init__()` | — | — | Starts one page, `y = PAGE_H - MARGIN`. |
| `content_width` (property) | — | `float` | `PAGE_W - 2*MARGIN`. |
| `new_page()` | — | — | Push a fresh page, reset `y`. |
| `_ensure(needed)` | `float` | — | Page-break if `needed` pt won't fit. |
| `spacer(height=8)` | `float` | — | Advance the cursor (vertical gap). |
| `_register_image(jpeg)` | `(bytes,w,h,comps)` | `str` name (`ImN`) | Registers an image XObject on the current page. |

Text primitives:

| Method | Input | Output | Purpose |
|---|---|---|---|
| `_draw_line_op(s, x, baseline, size, bold, color)` | one line + absolute coords | — (appends a content op) | Emits a `BT … Tj ET` text-showing operator. The low-level draw call every text helper funnels through. |
| `_wrap(s, size, bold, max_width)` | text, metrics, column width | `List[str]` | Greedy word-wrap; hard-breaks a single word wider than the column. Honors embedded newlines. |
| `text(s, *, size, bold, color, indent, leading, space_after)` | paragraph + style | — | Wrapped, auto-paged flowing paragraph. |
| `heading(s, *, size, color)` | title text | — | Bold `text()` wrapper. |
| `label_value(label, value, *, size)` | two strings | — | Compact `Label: value` row, bold label, hang-indented value. |
| `hr()` / `hairline()` | optional color | — | Full-width rule (`hr` heavier, `hairline` fine). |
| `link(text, url, *, size, indent)` | display text + URL | — | Blue underlined single-line link; registers a clickable `/Link` annotation. Truncates display text to column width; full URL still opens. |

Branded layout helpers (the report's visual identity):

| Method | Input | Output | Purpose |
|---|---|---|---|
| `_fill_round_rect(x, y, w, h, r, color)` | rect + corner radius + rgb | — | Bezier-rounded filled rectangle; the building block of the logo hand. |
| `_draw_noentry_emblem(cx, cy, box)` | center + box size | — | Draws the **warning-diamond logo**: yellow diamond, black border, white "stop" hand (palm + thumb + four fingers). |
| `_draw_wordmark(x, baseline, size)` | position + size | `float` (end x) | Draws `1-866 ` (ink) + `NO` (red) + `ENTRY` (ink). |
| `brand_header(*, company, tagline, right_text)` | strings | — | Top-of-page block: emblem (box=42) + wordmark + tagline + optional right-aligned meta (timestamp) + hairline. |
| `title_center(s, *, size, color)` | title | — | Centered report title. |
| `section_band(label, *, fill, text_color, size)` | label + colors | — | Full-width filled band — a section/record separator. |
| `field_label(label)` | string | — | Small uppercase field label (e.g. above the photo). |
| `field_row(label, value, *, size, link_url)` | label + value (+ URL) | — | The workhorse record row: small uppercase label, wrapped value (or clickable link), trailing hairline. Empty value → `—`. |
| `image_block(jpeg, *, max_height, link_url)` | `(bytes,w,h,comps)` | — | Flows a large event photo scaled to the column, optional click-through link. |

Serialization:

| Method | Input | Output | Purpose |
|---|---|---|---|
| `render()` | — | `bytes` | Assembles the full PDF: Catalog, Pages, two base-14 fonts (`/WinAnsiEncoding`, no embedding), one image XObject per snapshot (`DCTDecode`), FlateDecode-compressed page content streams, `Page N of M` footers, `/Link` annotation objects, and a cross-reference table. |

**PDF facts worth knowing:** fonts are Helvetica / Helvetica-Bold (base-14, so no
font file is embedded); color images are `/DeviceRGB`, grayscale `/DeviceGray`,
4-component `/DeviceCMYK`; raw JPEG bytes are stored verbatim behind the
`DCTDecode` filter; content streams are zlib-compressed. The logo is **drawn as
vector operators**, not embedded as a raster — there is no SVG rasterizer in the
environment.

---

### 2.2 `pdf_report_service.py` — `PdfReportGenerator`

The orchestrator. Instantiate per request (like a repository) or reuse a
singleton; only the injected `session_factory`, `email`, and dashboard URL are
state.

#### Constructor

`__init__(*, session_factory, email=None, dashboard_base_url=None)`
- **session_factory** — `async_sessionmaker[AsyncSession]`; opened with `async with … as db`.
- **email** — object exposing `send_report(...)`; `None` disables emailing (build still works).
- Reads env tunables: `REPORT_MAX_ALERTS` (500), `REPORT_MAX_IMAGES` (300), `REPORT_IMAGE_FETCH_CONCURRENCY` (8), `REPORT_IMAGE_FETCH_TIMEOUT_S` (15.0).

#### Public methods

**`generate_and_send(...) -> ReportResult`** — build + archive + email.

| Param | Type | Meaning |
|---|---|---|
| `org_id` | `int` | Target organization. |
| `report_type` | `"general"｜"urgent"` | Content-selection mode (below). |
| `start`, `end` | `datetime?` | Reporting window on `detected_at`. |
| `window_hours` | `int?` | If set and `start` omitted → `start = now - window_hours`. |
| `operator_approved_only` | `bool` | Drop rows whose `approved_by` is null. |
| `extra_recipients` | `list[str]?` | Merged on top of members + site recipients. |
| `notification_ids` | `list[int]?` | Urgent path: exactly these alerts. |
| `persist` | `bool` | Archive the PDF row. |
| `generated_by` / `generated_by_email` | `int?` / `str?` | Attribution for the archive + operator filter. |

Returns a `ReportResult`; sets `emailed` after attempting SMTP (no-op with clear log when `email` is `None` or no recipients resolved).

**`build_report(...) -> ReportResult`** — the core pipeline (same params minus `window_hours`; `persist` defaults `False`). Steps:
1. Open a session; load org, resolve **`prepared_for`** (org owner's display name), list org sites → name/location/timezone maps keyed by `site_uuid`, plus camera-location map.
2. Select `Notification` rows:
   - **urgent + `notification_ids`** → exactly those ids, scoped to org sites, `approval_status="approved"`.
   - **urgent (no ids)** → approved + visible + `emailed=True` in the window.
   - **general** → approved + visible + `emailed=False` in the window.
3. Resolve email recipients (members + enabled site notification emails + extras, de-duped case-insensitively).
4. Map each row → `_AlertEntry` (skip non-operator-approved when `operator_approved_only`).
5. For urgent-by-ids with no window, derive `[min, max]` from the alerts' `detected_at`.
6. `_attach_images()` — fetch + normalize snapshots (bounded concurrency).
7. Mint **one `report_uuid`** (`str(uuid.uuid4())`) shared by the printed "Report ID" **and** the archived row.
8. `_render_pdf(...)` → PDF bytes.
9. Build `ReportResult`; if `persist`, `_persist_report(...)`.

**Return type — `ReportResult` (dataclass):** `report_type, report_id (None until persisted), report_uuid, org_id, org_name, alert_count, image_count, site_count, start, end, recipients[], emailed, pdf_bytes, filename`.

#### Internal methods

| Method | Input | Output | Purpose |
|---|---|---|---|
| `_persist_report(*, result, entries, generated_by, generated_by_email)` | result + entries | `int?` (new row id) | Builds `ReportCreateDTO`, `ReportRepository.create`, commits. Swallows+logs errors → `None`. |
| `_entry_from_row(row, site_name…, site_loc…, site_tz…, cam_loc…)` | ORM notification + lookup maps | `_AlertEntry` | Pulls `title/body/classes/max_conf/notes/image_url/clip_url` out of `payload` (`msg`/`extra` sub-dicts); resolves site + camera names/locations. |
| `_resolve_owner_name(db, *, org, org_name)` | session + org | `str` | Org owner's `user_name`; falls back to org name. |
| `_extract_image_url(msg, extra)` / `_extract_clip_url(msg, extra)` | payload dicts | `str?` | First non-empty snapshot / recording URL (handles nested `extra["clip"]`). |
| `_resolve_recipients(db, *, org_id, site_uuids, extra)` | session + ids | `list[str]` | Members' emails + enabled site notification emails + extras, order-preserving de-dupe. |
| `_attach_images(entries)` | `list[_AlertEntry]` | — (mutates `.image_jpeg`) | Concurrent (`Semaphore`) snapshot fetch via one `httpx.AsyncClient`; caps at `REPORT_MAX_IMAGES`. |
| `_fetch_image_bytes(client, url)` | client + url | `bytes?` | Handles `data:` URLs (base64 or percent-encoded) and `http(s)`. Failures → `None` (logged). |
| `_render_pdf(*, report_uuid, report_type, prepared_for, entries, start, end, site_count)` | assembled data | `bytes` | Header + title + info block (**Report ID, Prepared for, window, sites, approved count — no org name**) + `OBSERVATIONS` band + one record per entry. Empty → "No alerts" note. |
| `_render_record(doc, idx, entry)` | doc + index + entry | — | One observation: numbered grey band + rows (Time in site tz, Site, Address, Camera, Type, Alert Message, Notes, Action Taken, Reviewed) + inline photo or link + clip link. |
| `_detection_text(entry)` | entry | `str` | `"<Alert type> · Detected: a, b · 87% confidence"`. |
| `_notes_text(entry)` | entry | `str` | Class-aware bulleted notes: **vehicle** → Type/Model/Color/Direction; **person** → Type/Gender/Clothing/Direction; **other** → Color/Direction; plus free-text `Note:` and `By:` author. Falls back to `Detected: …`. |
| `_actions_text(entry)` | entry | `str` | Bulleted `action` field(s) from the notes → the "Action Taken" row. |
| `_bullet` / `_primary_class` | — | helpers | Append `• Label: value` only when non-empty / titlecase first class. |
| `_subject` / `_email_text` / `_email_html` | result | `str` | SMTP subject + plain + HTML bodies. |
| `_fmt_dt` / `_fmt_dt_site` / `_fmt_window` | datetimes | `str` | UTC stamp / site-local stamp (`ZoneInfo`, UTC fallback) / window phrase. |
| `_filename(report_type, report_uuid)` (static) | strings | `str` | `noentry-{type}-report-{YYYY-MM-DD-HHMM}-{short-uuid}.pdf`. **No org name.** |

**`report_type` selection semantics:**
- `"urgent"` — alerts an operator approved **with** the email opt-in (already pushed to users); usually driven by explicit `notification_ids` from the approval route.
- `"general"` — approved alerts **not** emailed at approval; the daily roll-up.

---

### 2.3 `report_repository.py` — `ReportRepository`

Persistence for `organization_reports`. Stateless; **never commits** (caller
owns the transaction). Because the PDF blob can be multiple MB, list queries
select `_LIST_COLUMNS` (everything **except** `pdf_data`).

| Method | Input | Output | Notes |
|---|---|---|---|
| `create(db, *, dto: ReportCreateDTO)` | session + DTO | `OrganizationReport` | Inserts one row (`pdf_size = len(pdf_data)`), `flush` only. |
| `list_reports(db, *, org_id, report_type?, created_after?, created_before?, generated_by_email?, site_uuid?, report_uuid?, limit=100, offset=0)` | filters | `List[Row]` (blob-free) | Newest first. `report_uuid`/`report_type`/`created_at`/`generated_by_email` filter in SQL; `site_uuid` membership is tested in Python over the page (JSON list). `limit` clamped to [1, 500]. |
| `get_with_pdf(db, *, report_id, org_id=None)` | id (+org) | `OrganizationReport?` | Full row **including `pdf_data`**; org-scoped when `org_id` given. |

---

### 2.4 `report_scheduler.py` — `ReportScheduler`

Background loop that emails the **daily general report** per org.

| Member | Input | Output | Purpose |
|---|---|---|---|
| `resolve_tz(name)` (module fn) | tz name | tzinfo | `ZoneInfo`, UTC fallback. |
| `__init__(*, session_factory, generator_factory, poll_s=60)` | deps | — | `generator_factory` is a zero-arg callable → fresh `PdfReportGenerator`. `poll_s` floored to 15. |
| `start()` / `shutdown()` | — | — | Create / cancel the asyncio task. |
| `_loop()` | — | — | Every `poll_s`: `_tick()`, guarding `CancelledError`. |
| `_tick()` | — | — | Load enabled schedules; for each, compute local `today`, skip if already sent today or before send time; collect the due orgs. |
| `_send_for_org(*, org_id, window_hours, operator_only, today_str)` | — | — | **Stamps `last_sent_on` first** (so a slow/failed SMTP can't double-send that minute), then generates + emails. Failure is logged; retried next day. |

Idempotency: `last_sent_on` (a date string in the schedule's own timezone)
guarantees exactly one send per day even across restarts and minute polls. If
the process is down at the scheduled minute, it catches up on the next tick that
same day.

---

### 2.5 `report_routes.py` — HTTP endpoints

| Route | Method | Permission | Body/Query → Result |
|---|---|---|---|
| `/api/reports/approved-alerts` | POST | `REPORTS_SEND` | `ApprovedAlertsReportRequest` → **build + archive + email**; returns `ApprovedAlertsReportResponse`. |
| `/api/reports/approved-alerts/download` | POST | `REPORTS_SEND` | Same body → **build only**, returns the PDF `attachment` inline (no email/archive). |
| `/api/reports` | GET | `ORG_READ` | Filters `report_type, start, end, site_uuid, operator_email, report_uuid, limit, offset, org_id` → `List[ReportOut]` (blob-free). |
| `/api/reports/{id}/download` | GET | `ORG_READ` | Archived PDF, `Content-Disposition: attachment`. |
| `/api/reports/{id}/view` | GET | `ORG_READ` | Same PDF, `Content-Disposition: inline` — powers the in-app viewer (read without downloading). |
| `/api/reports/schedule` | GET/PUT | `ORG_MANAGE_SETTINGS` | Read / upsert the daily schedule. PUT pre-stamps `last_sent_on` if today's time already passed, so the first report is tomorrow. |

Org scoping: a normal caller's org comes from `OrgContext`; **platform admins**
(no single org) must pass `org_id`. Helpers: `_resolve_org_id`,
`_build_generator` (wires the email transport + dashboard URL from the
`NotificationService`), `_report_to_out`, `_load_report_pdf`.

---

## 3. Data model & contracts

**`OrganizationReport` (ORM, `organization_reports`):** `id` PK; **`report_uuid`
CHAR(36) unique+indexed** (stable shareable id, defaults to `uuid4`); `org_id`
FK; `report_type` (`general｜urgent`); `filename`; `generated_by` FK / `generated_by_email`
(indexed); `site_uuids` (JSON list); `period_start/end`; `alert_count`;
`pdf_data` LONGBLOB; `pdf_size`; `created_at`. Composite index
`(org_id, report_type, created_at)`.

**`ReportCreateDTO`** (repo input): `org_id, report_uuid, report_type, filename,
generated_by?, generated_by_email?, site_uuids[], period_start?, period_end?,
alert_count, pdf_data`.

**`ReportOut`** (HTTP response): `id, report_uuid, org_id, report_type,
filename, generated_by_email?, site_uuids[], period_start?, period_end?,
alert_count, pdf_size, created_at` — **metadata only, no blob**.

**Migration:** `create_all` never ALTERs existing tables. Because the
`mysql_data` Docker volume persists `organization_reports`, `core/database.py`
migrates existing DBs: add `report_uuid` nullable → backfill with `UUID()` →
add the unique index. New DBs get the column from the model directly.

---

## 4. Configuration (env)

| Var | Default | Effect |
|---|---|---|
| `REPORT_MAX_ALERTS` | 500 | Max notification rows per report. |
| `REPORT_MAX_IMAGES` | 300 | Max snapshots embedded (0 disables images). |
| `REPORT_IMAGE_FETCH_CONCURRENCY` | 8 | Parallel snapshot fetches. |
| `REPORT_IMAGE_FETCH_TIMEOUT_S` | 15.0 | Per-fetch httpx timeout. |

---

## 5. Extending it — common tasks

- **Add a record field** → add a `doc.field_row(...)` call in `_render_record`, and populate the value in `_entry_from_row`.
- **Change branding** → edit `BRAND_YELLOW/BRAND_RED/NAVY/ACCENT` and the `_draw_*` helpers in `pdf_builder.py`.
- **New report type** → extend the `report_type` normalization + row-selection branch in `build_report`, and `_REPORT_TYPE_LABELS`.
- **New archive filter** → add a param to `ReportRepository.list_reports` + the `/api/reports` query signature + `_report_to_out` if it surfaces a new field.
- **Embed the real logo raster** (instead of the vector emblem) → drop a JPEG, `normalize_to_jpeg` it, and call `image_block`/a new `_register_image` path in `brand_header`.
