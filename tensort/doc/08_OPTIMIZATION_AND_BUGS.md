# 08 — Optimization, cleanup, and existing bugs

This is a review report, not a patch set. It prioritizes the reported symptoms: late camera appearance and missing/delayed detections on an Orin Nano 8GB using YOLO26m.

Evidence labels: **reproduced** means an isolated execution or existing failing regression confirms it; **source-confirmed** means the control/data flow demonstrates it, without a live deployment reproduction; **measure** means a performance hypothesis requiring observations.

**Update 2026-09-22.** The deployed `Backend/tensort/.env` was supplied and reviewed. It differs materially from `.env.example`, and several of its values explain the reported symptoms directly, without any code change. Those are catalogued as **C1–C6** below and should be addressed *first*: they are configuration, they are reversible, and they are cheaper than the code fixes. The test suite was also executed on this checkout; results are recorded per finding.

Test run, `python3 tests/run_tests.py`, 2026-09-22: **16 suites passed, 2 failed** — `test_discovery.py` (2 failures, confirming **B2**) and `test_discovery_networks.py` (1 failure, confirming **B13**). Suites requiring CUDA/TensorRT hardware are not exercised on this host.

## Configuration findings — act on these first

These come from the deployed `.env`, not from the code. See [document 09](09_DISCOVERY_FLOW_END_TO_END.md) for the full value-by-value trace.

| ID | Finding | Why it matters | Fix |
| --- | --- | --- | --- |
| C1 | `GST_LATENCY_MS=4000` — a 4-second RTSP jitter buffer on every stream | Almost certainly the **largest single contributor to "detections are delayed"**. It is invisible to edge metrics because `ts_ms` is stamped *after* the buffer, so `last_frame_age_ms` can read 120 ms while the image is 4 s old | Measure true lag with a clock in frame; lower in steps and watch `gstreamer_failures` |
| C2 | `STATIC_NVRS` declares `1-6` for McCallum, but the comment beside it records 15 verified channels | Channels 7–15 are never candidates, never verified, never reach the cloud. Looks identical to "discovery is broken" | Decide the intended channel count; note 15+2 exceeds `MAX_CAMERAS=8` (see C3) |
| C3 | 8 declared channels against `MAX_CAMERAS=8` | Selection is permanently full. `devices[:8]` truncates **before** verification, in declaration order, so raising McCallum to `1-15` would silently drop **both** Penticton channels | Interleave declarations, or raise the limit only with measured GPU headroom |
| C4 | `DEFAULT_RESIZE_H=480` with `DEFAULT_RESIZE_W=640` | The GStreamer path applies this as a caps filter that does **not** preserve aspect ratio. A 16:9 substream is squeezed into 4:3, making people ~33 % narrower. Letterboxing later cannot undo it — it pads an already-distorted image. Costs accuracy on distant standing people | Set `DEFAULT_RESIZE_H=360` |
| C5 | `CAMERA_START_STAGGER_S=6` (default 1.5) | The connect gate is a *reservation*: each caller pushes the next slot +6 s whether or not it connects. Cold start pays 16 gate turns (8 verify + 8 adopt) = **96 s of pure waiting** before any handshake | Lower only after confirming the remote recorders tolerate faster session opens |
| C6 | `DISCOVERY_INTERVAL_S=240` | Sweep start-to-start ≈ `duration + 240 s`. With `DISCOVERY_MISS_THRESHOLD=2`, a dead camera takes **8–10 minutes** to be reported missing, and a returning one up to 4 minutes to recover | Tune against recorder session pressure, not in isolation |

C1 and C4 are the two worth changing today: one addresses the delay symptom, the other silently degrades detection quality on every frame.

## Priority overview

| ID | Priority | Finding | Evidence |
| --- | --- | --- | --- |
| B1 | P1 | Changed selected source can leave verified camera with no active capture | Reproduced |
| B2 | P1 | Dynamically absent cameras stop aging after one missed selection | Existing regression failures and source trace |
| B3 | ~~P1~~ **FIXED** | Manual inventory adoption loses edge UUID | Reproduced, then **fixed and regression-tested 2026-09-22** |
| B4 | P1 | Every edge upsert replaces capture, even when unchanged | Source-confirmed |
| B5 | P1 | Discovery-managed reconcile bypasses intended schedule removals | Source-confirmed |
| B6 | P2 | Late cloud cameras from long scan/quiet period/sequential work | Source-confirmed timing; deployment impact to measure |
| B7 | P1 | Cloud `dry_run` can still mutate MediaMTX/edge discovery state | Source-confirmed |
| B8 | P2 | Manual-mode replacement at capacity adds before removing | Source-confirmed |
| B9 | P2 | Cloud SSE shares a fixed connection pool with snapshot/latest reads | Source-confirmed; fleet-load impact to measure |
| B10 | P2 | Expired ROI cache can repeatedly block publication on database failure | Source-confirmed |
| B11 | P2 | Camera persistence errors hidden from successful mutation responses | Source-confirmed |
| B12 | P2 | Dormant notification buffer code has unsafe retry merge semantics | Source-confirmed cleanup debt; not active enqueue behavior |
| B13 | P2 | Static/local-discovered merge loses richer identity metadata | Existing regression failure; policy needs clarification |
| B14 | P2 | API-key behavior differs between CRUD and detection clients | Conditional deployment defect |
| B15 | P2 | Readiness does not verify running dispatcher/worker liveness | Source-confirmed |

P1 means fix before relying on the affected behavior in tower deployments. P2 means a consequential reliability/performance improvement. These are engineering priorities, not claims about the frequency of production incidents.

## B1 — Selection can retire a camera before source repointing

Sources: [`service.py`](../service.py), `_sweep`, `_inspect_candidate`, `_existing_camera_uuid_for`; [`runtime.py`](../runtime.py), `select_discovery_sources`, `_restore_cameras_from_db`.

Trigger example:

```text
Before: identity serial:S1 -> edge UUID A -> source rtsp://192.168.50.21/...
Next sweep: same serial S1 now appears at 192.168.50.22
```

The selection step removes A because its old URL is no longer selected. Saved configuration restore requires an exact selected URL, so A's old row is not restored. The new endpoint verifies. `mark_seen()` retains the roster's existing `camera_uuid=A`. But there is no active A to patch, and `needs_adoption` is false because the row is neither new nor unlinked. The sweep ends with a verified roster row and zero active captures for it.

An isolated reproduction using actual runtime/service methods and fake repositories confirmed this. Credential changes and stream/offset changes can exercise the same exact-URL transition even without DHCP. Deleting a camera configuration while leaving a linked roster row creates a related recovery hole.

**Fix direction:** reconcile identities against persisted and active configurations before retiring sources. Treat “linked UUID is not active/restorable” as a recovery state. Preserve UUID, replace its configuration once, and attach/update roster linkage only after successful admission/persistence. Avoid minting a replacement UUID merely because the old channel was removed from memory.

**Acceptance:** with discovery management enabled, move a serial-identified camera's IP and rotate its credentials/path in separate cases; the same UUID resumes capture, old capture is stopped, one configuration remains, and cloud UUID/subscription remains valid. Test removal/return and restart between steps.

## B2 — Dynamic disappearance does not reach missing threshold

Source: [`DiscoveryService._sweep`](../service.py), its `excluded` identities and `mark_missing` input.

```text
Sweep 0: selected={A}, A present
Sweep 1: selected={}, previous={A}; A gets consecutive_misses=1
Sweep 2: selected={}, previous={}; A classified as excluded historical row
         excluded is supplied as present to mark_missing; A stays at 1 miss
```

With the default threshold 2, a camera that disappears from the dynamic candidate list can fail to become missing. It consequently also fails to produce the intended recovery transition.

**Reproduced on this checkout, 2026-09-22.** Running `tests/test_discovery.py` gives:

```text
FAIL: test_missing_respects_grace_window_and_alerts_once   AssertionError: 0 != 1
FAIL: test_recovery_after_missing                          AssertionError: 0 != 1
```

Static candidates remaining in selection while frame verification fails are different: they are still tested and can reach the threshold. Do not generalize the defect to every kind of offline channel.

**Scope on the current tower.** With `STATIC_NVRS` and `DISCOVERY_LOCAL_ENABLED=false`, the candidate set is rebuilt identically from `.env` every sweep and never varies. A channel that merely fails verification therefore stays in `selected_identities`, is correctly absent from `present_identities`, and **does** age normally. The defect bites when the candidate set itself changes — editing `STATIC_NVRS`, or enabling ISAPI/local discovery. Fix it before relying on missing-camera alerts, but do not expect it to explain a missing camera today.

**Fix direction:** persist or retain the distinction between explicitly excluded/decommissioned identities and previously managed identities that are currently absent. Keep aging absent managed identities across sweeps. Never age untested candidates because a sweep was cancelled.

**Acceptance:** two absent complete sweeps produce exactly one missing transition; another absent sweep does not repeat it; return produces one recovered transition. Repeat with a process restart and with static, local-camera, and NVR identities.

## B3 — Inventory-add drops the UUID already used by the edge ✅ FIXED

Sources: [`InventoryService._adoption_entry`](../../application/services/inventory_service.py), [`CameraAdopter._create_camera`](../../application/services/manager/controllers/adopt.py), [`CameraInventory.edge_camera_uuid`](../../core/database_orm.py), edge `_apply_camera`.

### The defect

Inventory stored `edge_camera_uuid`, but `_adoption_entry()` omitted it, returning exactly six keys. `_create_camera()` then minted a fresh UUID:

```python
camera_uuid = entry.get("camera_uuid")
cam_uuid = uuid.UUID(str(camera_uuid)) if camera_uuid else uuid.uuid4()
```

```text
Edge selected source -> UUID A
Manual inventory add -> new cloud UUID B
Cloud POST source with B -> edge rejects: selected source already has a camera UUID
Cloud subscribes to B -> edge emits only A -> no matching detections
```

The camera showed live video (MediaMTX pulls the RTSP source directly and never needs the UUID to agree) while producing **no detections at all**. The automatic reconcile adoption path always passed the roster's UUID and never had this omission — only the "add to site" screen did.

### The fix, applied 2026-09-22

`_adoption_entry()` now carries the stored value through, defensively:

```python
edge_uuid = str(row.edge_camera_uuid or "").strip()
if edge_uuid:
    try:
        entry["camera_uuid"] = str(uuid.UUID(edge_uuid))
    except ValueError:
        logger.warning(...)   # drop it; adopting under a new UUID beats no camera
```

`edge_camera_uuid` is a free-text `String(64)`, so a malformed value is dropped with a warning rather than raising — `uuid.UUID(...)` in the adopter would otherwise fail the entire add. A `NULL`/blank value omits the key, preserving the previous mint-a-new-one behaviour. Existing cloud rows are safe because the channel-create path is an **upsert** (`upsert_camera_from_channel_config`), so reusing a UUID updates in place instead of colliding.

### Verification

Five regression tests were added as `AdoptionEntryTests` in [`tests/test_camera_inventory.py`](../../tests/test_camera_inventory.py), covering: valid UUID carried through, spelling normalised (braced/uppercase forms), `None`/empty/whitespace omitted, malformed dropped without raising, and other reported fields preserved.

```text
new tests vs the OLD code:  2 failed, 3 passed   <- the tests do catch the bug
new tests vs the FIXED code: 5 passed
full backend suite:         205 passed
```

The two failures in `tests/test_tensort_pipeline_autosizing.py` are **pre-existing and unrelated** — confirmed by re-running them with this change reverted. Their `_FakeInferenceWorkerPool` double requires a positional `num_workers` that `tensort/pipeline.py` no longer passes.

### Still open — find the cameras already broken

This fix restores UUID continuity for **new** adds. It does not migrate cameras already filed under a mismatched UUID by the old code. Those cameras still show live video and still produce no detections, so they will not surface on their own.

A read-only audit ships for exactly this:

```bash
python3 Backend/scripts/audit_inventory_camera_uuids.py          # report
python3 Backend/scripts/audit_inventory_camera_uuids.py --json   # machine-readable
```

It compares every `state='added'` inventory row's `edge_camera_uuid` against the `camera_uuid` of the camera row it points at, and sorts the results into **mismatched** (broken), **unparseable**, **dangling**, **no edge UUID** (legacy, usually fine) and **ok**. Exit status is `1` when mismatches exist and `0` when clean, so it can gate a deploy.

The comparison runs through the ORM deliberately: on MySQL `camera.camera_uuid` is `BINARY(16)` while `camera_inventory.edge_camera_uuid` is text, so a direct SQL join between the two columns would silently match nothing and report a false all-clear.

The script does not repair anything — re-pointing a camera means updating every table that references its UUID and changes `camera_code`, and therefore the MediaMTX path. It prints the two remediation options (delete-and-re-add, or in-place re-point preserving ROI/alert settings) and leaves the choice to an operator. Legacy rows with no `edge_camera_uuid` still resolve by minting, which remains correct.

## B4 — Unchanged upserts restart video and clear results

Sources: [`PipelineRuntime.add_camera/_apply_camera`](../runtime.py), [`SimpleInferencePipeline.add_channel`](../pipeline.py), [`ChannelController.add_channel`](../../application/services/manager/controllers/channel.py).

Cloud adoption schedules an upsert even when the edge already started that UUID. Edge `add_camera()` reconstructs defaults plus supplied fields, then always calls `pipeline.add_channel()`. The pipeline stops the old channel, changes generation, clears results/frame pool, and starts a new capture. A small metadata update can therefore incur connection staggering, RTSP handshake, keyframe wait, and an overlay gap. A sparse POST can also replace saved per-camera tuning with defaults because it is not a PATCH merge.

**Fix direction:** compare effective capture configuration, separate metadata-only updates from capture changes, and make equal upserts no-ops at the capture layer. Define POST replacement versus PATCH semantics explicitly. Commit state changes coherently and expose whether an operation restarted capture.

**Acceptance:** repeat the same POST ten times; capture thread/generation and advancing sequence continue. Change notification metadata without restarting. Change source/codec-related configuration once and observe one deliberate replacement with stale-generation protection retained.

## B5 — Schedule policy and discovery selection disagree

Sources: cloud [`DeviceReconciler._recompute_desired/reconcile_device_edge_simple`](../../application/services/manager/controllers/reconcile.py), edge discovery `_adopt`.

The reconciler computes desired detection cameras using detection flags and the site's armed state. Later, when health reports discovery-managed selection, it clears both `to_add` and `to_remove`. The edge auto-adopts cameras enabled for detection independently of site schedules. The discovered-UUID protection also classifies some known-but-undesired cameras as “unadopted.”

Therefore the comments promising that a disarmed site's edge capture is torn down do not hold for this path. Cloud notification gating still applies, so this can present as “no alerts” while the Jetson continues consuming resources.

**Fix direction:** separate physical source membership, capture enablement, inference enablement, and alert authorization. Let discovery own identity/source discovery while a clearly defined desired policy controls inference for existing selected UUIDs. Do not solve it with delete/re-add loops.

**Acceptance:** arm/disarm and schedule boundaries change the intended inference state on a discovery-managed tower while preserving UUID, discovery history, and required live playback. Specify whether idle capture should remain warm or be closed.

## B6 — Registration latency is cumulative, not just the scan interval

Sources: edge scanner/service/channel and cloud reconciler/adopter, frontend refresh.

Confirmed contributors include:

- A 150-second quiet period, absent from `.env.example`, measured after sweep completion.
- Full local scan completion before verifying even static candidates.
- Sequential configured-recorder probes and sequential selected-candidate video verification.
- Up to six GStreamer attempts plus FFmpeg for a failing selected source.
- Temporary verification connection followed by a second persistent connection.
- Global 1.5-second connection gate shared by independent devices and probes.
- Cloud sequential device reconciliation and MediaMTX work before adoption.
- Browser camera-list refresh on a 30-second timer.

**Fix direction:** expose a progress state per candidate and stage duration; return verified partial state consistently; use a persistent bounded selection policy; split inventory observation from expensive video validation; reuse a verified capture when ownership transfer can be made safe. Probe unrelated hosts with bounded concurrency while retaining a small **per-recorder** connection limit/backoff. Make quiet-period/cadence values visible in the shipped configuration and status UI.

Do not parallelize every channel of one NVR or remove connection gating indiscriminately. Session storms can lengthen recovery more than serial probing does. Selection-before-verification intentionally reserves slots for offline selected sources; changing it requires an explicit failover/admission policy.

**Acceptance:** instrument time to first selected camera, all verified cameras, cloud rows, and first cloud results for direct/NVR/mixed topologies, including dead early candidates. Compare percentiles before/after under the same network conditions.

## B7 — `dry_run` is not free of side effects

Source: [`reconcile_device_edge_simple`](../../application/services/manager/controllers/reconcile.py).

MediaMTX `ensure_stream` and, when allowed, `delete_stream` execute before the final `if dry_run: return out`. The function also calls edge `/sync`, which can trigger an adopting discovery sweep. The adopter's own dry-run checks do not undo these earlier side effects.

**Fix direction:** separate observation, plan calculation, and application. A preview should use read-only report/list endpoints and make no external mutations, database changes, or background jobs. If an explicit “refresh inventory while planning” mode is wanted, name and document it separately.

**Acceptance:** mocked edge/media/database clients record zero mutations for a dry-run, including a missing media path and a newly discovered source. No network discovery is scheduled by a read-only preview.

## B8 — At-capacity replacement order is inverted

Source: same reconciler, capacity calculation and final add/delete loops.

Manual-mode example: edge contains eight cameras, one should be removed, and one new camera should be added. The capacity calculation counts the planned removal as a free slot. But the execution loop posts additions **before** deletions. The edge correctly rejects the addition with 409 while still full; the deletion happens later, so addition waits for another pass.

**Fix direction:** plan an explicit replacement order or atomic replace operation. Validate the new configuration before releasing the old slot; then remove/replace and add within a controlled recovery workflow. Avoid breaking working cameras just to make a speculative addition fit.

**Acceptance:** at-capacity one-for-one replacement succeeds in one apply pass, with well-defined rollback/retry if the replacement source fails. Include disabled cameras in the policy tests so counts reflect enabled admission correctly.

## B9 — Shared HTTP pool and old SSE backlogs

Sources: cloud [`application/channels/channel.py`](../../application/channels/channel.py), edge `Broadcaster` in [`pipeline.py`](../pipeline.py).

One long-lived cloud SSE per camera shares a 100-connection HTTPX client with latest/snapshot requests. At fleet scale the streams can occupy the pool, delaying short requests or preventing later subscriptions. The client's lifetime is module-scoped rather than clearly owned by the FastAPI lifespan.

Each edge subscriber can queue 200 events. At 3 FPS that can represent about 66.7 seconds for a camera, and drop-oldest only occurs once full. There is no explicit age limit on subscriber events. This is bounded memory, not guaranteed fresh delivery.

**Fix direction:** separate streaming and short-request clients, close them in lifecycle cleanup, instrument pool wait/reconnects, and consider one device SSE with demultiplexing. For live overlays, use a small latest-value/coalescing buffer or explicit event age policy. Keep tracking needs in mind: discarding too aggressively changes missed-object evidence.

**Acceptance:** load-test more than 100 camera subscriptions plus snapshots, transient WAN stalls, and reconnect. Report per-camera first event and event-age percentiles, not only successful socket counts.

## B10 — ROI lookup remains before live publication

Source: cloud [`ModelPipeline._process_detection_payload/_fetch_rois`](../../application/services/pipeline.py).

A cache miss/expiry awaits SQL before storing/publishing the response. On a database exception with an old cache entry, `_fetch_rois()` returns the old ROI but does not extend its expiry. Every following tracked frame can retry the same failing query, with pool waits delaying detection consumption.

**Fix direction:** use a bounded stale-while-refresh cache with a per-camera refresh task and retry deadline; optionally publish raw detection/tracking before attaching asynchronously computed alert state, with clear ordering. Preserve correct ROI invalidation when edited.

**Acceptance:** inject a slow/down database while valid cached ROI exists. Overlays continue; at most one refresh per camera is in flight; retries are bounded; restored SQL updates the cache; ROI edit invalidation still works.

## B11 — Mutation success can hide persistence failure

Sources: edge [`camera_repository.py`](../repositories/camera_repository.py), runtime `_apply_camera`, `_call`.

The runtime changes live capture before saving. Save/delete exceptions are logged and swallowed. A 201/success response can represent only an in-memory change, which disappears after reboot, or a delete that reappears. `_call()` also leaves a coroutine running when its blocking future wait times out; HTTP failure is not proof that no mutation occurred.

**Fix direction:** give repository failures explicit results/exceptions; define persistence and runtime transition order, compensation, and retry idempotency. On timeout, expose an operation state or use deliberate cancellation semantics where safe, rather than implying completed rollback.

**Acceptance:** inject SQLite write failure and delayed coroutine completion. Response/result state accurately describes persistence and runtime state; restart does not silently reverse an operation reported as durably successful.

## B12 — Dormant notification buffer needs removal or repair

Source: [`NotificationFlusher.enqueue/_flush_ready_users`](../../application/services/notification/flusher.py).

A second-pass caller trace found that the current `enqueue()` bypasses `_pending_by_user`: it prepares, flushes, and commits immediately. Retained 100-item/60-second buffer settings and the background flush loop are therefore misleading for the active path, and should not be blamed for production delay without evidence of another producer.

If that dormant buffer is reactivated, its failure branch assigns the failed batch over the current pending user batch instead of merging. An older failed batch A would replace a newer concurrently enqueued batch B. No current producer into that buffer was found in the reviewed source; this is a latent defect, not a demonstrated active alert-loss incident.

**Fix direction:** remove obsolete buffer machinery/configuration if immediate persistence is the intended contract. If batching is restored, merge retries under the lock, preserve earliest age and IDs, and define durable retry/backpressure behavior. The immediate path also needs a deliberate retry policy: currently a false flush result rolls back, logs, and returns rather than using the old queue.

**Acceptance:** preserve the existing immediate-commit regression. If a buffer is introduced, pause a failing flush after it removes A, enqueue B, resume failure, and assert both remain exactly once. Document which latency/retry settings actually apply.

## B13 — Static versus discovered metadata contract is inconsistent

Source: [`scan_network`](../discovery.py); failing test `test_static_and_isapi_versions_of_same_channel_do_not_duplicate` in [`test_discovery_networks.py`](../tests/test_discovery_networks.py).

Static candidates are enriched using the configured/dynamic-NVR scan, but not the later local scan. If a duplicate local endpoint carries serial/model metadata, endpoint deduplication preserves the earlier static record. The failing test expects richer local records; the actual result retains static identity/model data. It does **not** show two simultaneous duplicate endpoint captures.

Normal local scanning excludes explicitly configured static hosts, so the particular mocked test combination is not proof of a frequent production path. Still, metadata precedence and identity migration need a documented policy: adding/removing configured ISAPI enumeration can switch from IP-based identity to serial-based identity.

**Fix direction:** define stable endpoint/identity aliases and a metadata merge independent of priority order. Update the test only after deciding the contract; do not suppress the failure as if semantics were already settled.

## B14 — Authentication and configuration hygiene

The cloud CRUD edge client sends `x-api-key` when configured; cloud SSE/latest/snapshot calls do not. Behind a key-enforcing gateway, successful camera listing with missing detections is therefore possible. Centralize the edge authentication contract and test all three data endpoints behind a fake authenticated gateway.

The edge Flask routes currently expose camera source URLs and mutation endpoints without their own authentication middleware. `Backend/docker-compose.yml` contains credential-shaped values, and the media gateway client sets `verify=False`. These are source-confirmed deployment/security cleanup items; the validity/exposure of any credential was not tested. Move secret material to deployment secret injection, review/rotate any real exposed credentials through the normal operational process, and establish authenticated transport. Do not copy existing values into documentation or diagnostics.

## B15 — Green readiness can survive a dead hot-path task

Sources: edge `_pump_inference`, `_pump_channel`, `peek_stats`, and route `runtime_status`.

Unexpected pump exceptions are logged and the task exits. Readiness checks `_started`, engine batch availability, and closing state, not task/worker liveness or recent progress. A stalled or exited pipeline can keep `pipeline_ready=true`.

**Fix direction:** separate process liveness, engine readiness, worker/dispatcher liveness, capture freshness, and per-camera inference freshness. Supervise recoverable tasks with bounded restart or fail the service explicitly when the engine worker dies. An offline camera should not necessarily make the entire process unready.

**Acceptance:** deliberately fail the dispatcher/worker and verify health reflects it; stop only one camera and verify others keep working while that camera reports degradation.

## Orin optimization plan, in order

### 1. Fix correctness and observe each stage

Address B1–B5 before benchmarking unexplained missing cameras. Add stage timestamps or histograms for discovery, connect, pool wait, worker queue wait, preprocessing, GPU execution, postprocessing, edge emission, cloud receive/publish, and browser arrival. Use monotonic clocks for within-process durations and synchronized wall clocks for cross-host correlation. Include camera UUID, generation/boot ID, and batch size without exposing credentials.

### 2. Reduce unnecessary video work

Measure the real substream's resolution/FPS/codec/bitrate and negotiated decoder. Prefer a camera-side profile that preserves required small-object detail with manageable decode/network cost. Retain hardware decode where supported. Cache a successful decoder strategy per endpoint so reconnect does not always repeat codec guesses; invalidate on stream changes and retain fallback. Use per-recorder connection budgets, not unconstrained parallel opens.

### 3. Match input load to measured YOLO26m capacity

Keep `yolo26m` as the baseline. Benchmark batch sizes 1, 2, 4, and 8 with representative images and full live capture. Target sustained per-camera output and latency under thermal steady state. Do not present the `.env.example` estimate of 30–45 images/s as an Orin measurement. Select the optimum profile from actual batch distribution; a maximum batch of eight need not be the most frequent batch on a four-camera tower.

If load exceeds capacity, reduce source/sample rate or adjust model input resolution with validation and an engine rebuild. Smaller model variants or INT8 are later experiments requiring labeled accuracy checks, especially for distant people/vehicles and night scenes. No claimed speedup is provided without a target measurement.

### 4. Keep bounded freshness policies

Preserve the coalesced handoff, fair pool, generation checks, one CUDA owner, and bounded JPEG executor. Evaluate whether the one queued batch materially improves GPU utilization relative to added age. Consider checking queued-batch age immediately before execution; preserve callback/future cleanup for dropped jobs. Increasing `FRAME_POOL_CAP`, worker queues, and timeouts together is likely to conceal overload with latency.

### 5. Profile copies before attempting zero-copy

Current improvements already include reusable pinned input/output buffers and batch parsing before reuse. Remaining copies include native appsink→owned BGR, resize/padding, CHW conversion, explicit host/device transfers, and JPEG encoding. Profile each first. GPU preprocessing or a native multimedia/GPU path can help, but requires explicit pitch/format/lifetime/context handling and may add complexity. Removing the safe native `.copy()` alone is a use-after-release risk, not a valid optimization.

### 6. Bound cloud work as carefully as edge work

`_post_publish_work` spawns per result. Tracking task references prevents tasks being lost to garbage collection; it does not impose a concurrency bound. Under slow database/storage/notification operations, outstanding work can grow. Use bounded queues/coalescing for playback metadata and deliberate durable handling for alerts. Isolate ROI refresh and HTTP clients; budget database connections across workers and replicas.

## Cleanup work with concrete outcomes

| Cleanup | Reason / desired result |
| --- | --- |
| Update README/route docstrings for selection and partial reports | Current wording describes restore-before-discovery/full historical rosters where current code does otherwise |
| Add quiet period, probe budget, reconnect tolerance to `.env.example` | Operators can see delays that currently hide behind defaults |
| Unify deployment/model defaults | Setup comments mention nano/2048 MB/opt 6 while executable defaults use yolo26m/3072 MB/opt=max; deployment README still has older tuning |
| Replace the “throughput gate” name or implement a real gate | Build script prints benchmark text and uses `|| true`; it does not enforce a required FPS threshold |
| Detect ONNX input name in both build paths | `build_engine.sh` checks/reports input name but shape flags hardcode `images`; setup script uses detected name |
| Remove obsolete Python 3.6/TensorRT 8 compatibility claims | Current runtime/requirements target a different stack |
| Rename historical `YoloV8DetTRT` / `infer_multitask_batch` thoughtfully | Current facade includes YOLO26 and produces `pose=None`; retain compatibility aliases if used elsewhere |
| Extract typed camera identity/source contracts shared conceptually across services | Avoid three subtly different matching/adoption implementations |
| Centralize validation with explicit finite/range checks | Some modules use safe environment parsers while others call raw `int/float`; malformed settings behave inconsistently |
| Separate reporting from mutable internal dictionaries | Lock-free single-key access does not make a compound snapshot transactionally consistent |
| Own clients/executors/engines in application lifecycle | Predictable closure and easier isolated tests |
| Consolidate historical root documentation | Many prior “fix complete” documents may describe older code; link dated evidence instead of treating them as current truth |

## Required checks before implementing a performance change

Compare same-camera recorded scenes and live mixed-topology operation, including empty frames, overlapping vehicles, small/distant people, night lighting, and intermittent network loss. Record delivered FPS, first-detection/recovery time, p50/p95/p99 age, memory, CPU/GPU utilization, thermal state, false positives/negatives, UUID continuity, and cloud notification outcomes. Keep engine/config versions with results. A faster synthetic TensorRT invocation alone does not establish a better tower.
