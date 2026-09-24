# 04 — YOLO, TensorRT, and CUDA with numerical examples

## What each library does

| Component | Job in this service |
| --- | --- |
| YOLO26m | Neural-network architecture/weights used to predict objects |
| ONNX file | Exported model graph used as input to an engine build |
| TensorRT engine | Compiled inference artifact for the deployment stack/profile |
| TensorRT execution context | Mutable inference state and actual runtime shapes |
| CUDA context | GPU resource/execution environment associated with the worker thread |
| CUDA stream | Ordered queue of GPU copies and inference operations |
| PyCUDA | Python access to contexts, streams, allocations, and transfers |
| NumPy/OpenCV | CPU image arrays, resize/padding, tensor preparation, and postprocessing |

There is no PyTorch training loop or Ultralytics Python prediction call in the edge hot path. The service loads a `.engine` file and runs its custom parser. `YoloV8DetTRT` is a historical class name: it also handles the six-column end-to-end output used for YOLO26 detection.

Ultralytics documents YOLO26's native end-to-end, NMS-free export behavior. The exact graph and output shape still depend on export choices; inspect the deployed artifact rather than inferring its interface from the filename. See [Ultralytics end-to-end detection](https://docs.ultralytics.com/guides/end2end-detection).

## Step 1: a decoded BGR image

Use one reproducible teaching case throughout:

```text
Camera UUID: 11111111-1111-4111-8111-111111111111
frame_seq: 42
frame_ts_ms: 1800000000000 (illustrative)
Input image shape: (360, 640, 3)
Input dtype: uint8, channel order BGR, values 0..255
IMG_SZ: 640
```

NumPy image indexing is `[y, x, channel]`. A pixel `[10,20,200]` means blue=10, green=20, red=200. The model input uses RGB instead.

`VideoChannel` has already decoded and possibly resized this image. A 640×360 frame does not preserve the pixel detail of a 1920×1080 original; inference cannot recover distant objects erased by that resize.

## Step 2: letterbox to square input

[`letterbox_bgr()`](../trt_infer.py) preserves the incoming image's aspect ratio and pads it to a square:

```python
r = min(640 / 360, 640 / 640)  # 1.0
nh = round(360 * r)            # 360
nw = round(640 * r)            # 640
pad_w = 640 - nw               # 0
pad_h = 640 - nh               # 280
left = 0
top = 140
```

The resulting BGR image is `(640,640,3)`. The top 140 rows and bottom 140 rows are padding filled with value 114 in each channel. Original pixel `(x=100,y=50)` appears at model pixel `(100,190)`.

For an unresized 1920×1080 input, `r=1/3`, resized width=640, resized height=360, and padding is again `(left=0,top=140)`. In that case reversing letterbox divides coordinates by `1/3`. In the actual 640×360 example `r=1`, so no coordinate scaling is required after subtracting padding.

## Step 3: fill the pinned input tensor

`_fill_row()` uses views to reverse BGR→RGB and rearrange HWC→CHW, then copies into the reusable pinned input row and normalizes it:

```python
np.copyto(row_chw, img_lb[:, :, ::-1].transpose(2, 0, 1), casting="unsafe")
row_chw *= 1.0 / 255.0
```

```text
Before: one image, (H,W,C) = (640,640,3), uint8
After:  one tensor row, (C,H,W) = (3,640,640), float32 or float16
Batch:  (B,C,H,W), e.g. (4,3,640,640)
```

For pixel BGR `[10,20,200]`, RGB channels become approximately `[0.784314,0.078431,0.039216]`. Padding value 114 becomes approximately `0.447059`. This is a normalized intensity, not a confidence score.

The “direct into pinned buffer” optimization avoids building and concatenating one floating-point blob per image. It does not eliminate all temporaries: `cv2.resize` and `copyMakeBorder` still create image arrays, and the final normalization touches the tensor row.

The code reads the engine's tensor dtype. An FP16 engine build can still expose FP32 input/output bindings. Do not assume `--fp16` means every host buffer uses two-byte values.

## Step 4: batch profiles and allocations

[`TRTEngine.__init__`](../trt_infer.py) uses TensorRT 10's tensor-name API. It finds one four-dimensional image input, its dtype, output tensors, and shape profile. A dynamic profile must accept batch 1 so partial camera batches can run.

Illustrative engine profile:

```text
input_name = "images"
declared_shape = (-1, 3, 640, 640)
minimum = (1, 3, 640, 640)
optimum = (8, 3, 640, 640)
maximum = (8, 3, 640, 640)
```

`-1` means a dimension will be supplied at runtime. If only three fresh frames are ready, the context uses `(3,3,640,640)`. The code does not wait for five silent cameras. Batch 8 is an upper limit; optimum batch 8 tells the builder which shape to optimize around.

The constructor allocates host and device buffers for the **maximum** profile size and binds their addresses once. Each call transfers only the active rows. It asks the execution context for resolved output shapes rather than guessing that every dynamic output dimension is the batch dimension.

A fixed batch=1 engine is supported. A fixed batch greater than 1 is rejected because the pipeline needs to run partial batches. A dynamic engine whose minimum batch is not 1 is rejected. `IMG_SZ` must match the supported spatial dimensions.

These concepts match [TensorRT 10's Python execution interface](https://docs.nvidia.com/deeplearning/tensorrt/10.x.x/inference-library/python-api-docs.html). The source remains authoritative for the additional restrictions imposed by this application.

## Step 5: what CUDA actually does

The worker creates a CUDA context using `cuda.Device(device_id).make_context()`. Thread-local storage retains it. `CudaContext` pushes it before GPU work and pops it afterwards. The same worker eventually frees allocations and detaches its context.

One batch follows this sequence:

```text
CPU: letterbox/normalize B images into pinned host input
CUDA stream: copy active input rows host -> device
CUDA stream: execute TensorRT network
CUDA stream: copy output tensors device -> host
CPU: wait at stream.synchronize()
CPU: parse outputs, apply filtering, build JSON dictionaries
```

`execute_async_v3()` queues GPU execution asynchronously relative to the calling CPU. However, this service then calls `stream.synchronize()` before reading outputs. Within this single worker's batch loop, it does not preprocess the next batch while this call is waiting. Capture and other threads can continue independently.

`copy_outputs=False` returns views into the reusable pinned output buffers. `run_batch()` parses those outputs before the next inference overwrites them. Retaining such an output view across calls without copying would read overwritten data.

On a Jetson, the CPU and integrated GPU share physical DRAM. The 8GB budget is not 8GB of CPU RAM plus another 8GB of discrete GPU VRAM. Nevertheless, this implementation explicitly allocates host/device buffers and issues copies; shared physical memory does not automatically make its NumPy/GStreamer path zero-copy. See [NVIDIA's CUDA for Tegra memory discussion](https://docs.nvidia.com/cuda/archive/13.0.2/cuda-for-tegra-appnote/index.html). This reference explains memory concepts, not the unknown installed CUDA version.

Pinned/page-locked host memory is allocated with `cuda.pagelocked_empty()`. It supports the explicit asynchronous transfer path. It is not the same concept as a database row lock, the Python GIL, or CUDA shared memory inside a kernel.

## Step 6: output decoding

The parser supports two layouts:

| Layout | Interpretation | Application postprocessing |
| --- | --- | --- |
| `(B,N,6)` with `N != 6` | `x1,y1,x2,y2,score,class_id` | Confidence/class filter, undo letterbox, clamp, dedupe/top-k |
| Raw YOLO class-score layout, commonly `(B,84,8400)` for 80 classes at 640 | Four box values plus per-class scores | Best class, confidence/class filter, xywh→xyxy, Python NMS, undo letterbox |

These are supported shapes, not confirmation of your `.engine` output. The code takes the first output tensor. Multi-output detection engines with separate count/box/score tensors need a different parser. The shape heuristic has limitations for ambiguous/custom shapes.

For YOLO26's six-column branch, the Python `IOU` NMS threshold is not used by `nms_xyxy`; the separate `DEDUPE_*` stage can still remove overlapping confusable boxes. The historical comment “end-to-end NMS output” should not be interpreted as proof an NMS plugin is embedded in the graph.

## A complete detection example

Suppose the model predicts this illustrative six-column row:

```text
[100, 190, 300, 390, 0.83, 0]
  x1   y1   x2   y2  score class_id
```

With `CONF=0.18` and allowed classes containing `person`, class ID 0 survives. Undo padding `(0,140)` and scale `r=1`:

```text
Original-frame box = [100, 50, 300, 250]
frame_w = 640, frame_h = 360
Normalized x = 100/640 = 0.15625
Normalized y = 50/360 ≈ 0.138889
Normalized w = 200/640 = 0.3125
Normalized h = 200/360 ≈ 0.555556
```

The example result looks like:

```json
{
  "type": "DetectionsProducedEvent",
  "channel_id": "11111111-1111-4111-8111-111111111111",
  "camera_uuid": "11111111-1111-4111-8111-111111111111",
  "model_id": "yolo-trt",
  "frame_ts_ms": 1800000000000,
  "frame_seq": 42,
  "frame_w": 640,
  "frame_h": 360,
  "detections": [{
    "cls_name": "person",
    "conf": 0.83,
    "box": {"x1": 100, "y1": 50, "x2": 300, "y2": 250},
    "box_norm": {"x": 0.15625, "y": 0.138889, "w": 0.3125, "h": 0.555556}
  }],
  "pose": null,
  "inference_ms": 240,
  "batch_size": 4
}
```

`240 ms` is an invented teaching value. In this code, `inference_ms` includes the batch's preprocessing, execution/copies, and parsing. Every frame in that batch gets the same batch wall time. It is not GPU-only time, queue delay, or a per-frame latency obtained by dividing by four. `model_id='yolo-trt'` is also not proof that the file loaded was YOLO26m; the configured artifact must be checked separately.

## Confidence, NMS, and duplicate suppression

Confidence is the model's score, not a calibrated guarantee that an object is present. `ALLOWED_CLASSES` maps through a fixed COCO-80 name table. A custom-trained class ordering would need an explicit mapping; otherwise names/filtering can be incorrect.

For two boxes A and B, IoU is `intersection area / union area`. If each has area 100 and their overlap is 80, IoU is `80/(100+100-80)=0.6667`. Duplicate suppression can keep the stronger box. The custom cross-class logic groups car/truck/bus/van as confusable vehicles; unrelated classes such as person/car are preserved. A motorcycle is not included in that vehicle confusion set.

A same-object car box at 0.82 and nearly identical truck box at 0.61 can become one car result. But two genuinely overlapping similarly sized vehicles may also be affected. Tune using labeled crowded scenes, not only an empty parking lot.

For cloud tracking defaults, scores ≥0.5 start new tracks; scores 0.20–0.5 can rescue existing tracks. With edge `CONF=0.18`, scores below 0.20 may appear as raw detections but do not enter those association bands. Keeping `CONF <= low_th` preserves the entire low-confidence band. A threshold above `low_th` truncates that band; it does not make the whole band empty unless it reaches/exceeds the high threshold.

## Memory arithmetic for Orin Nano 8GB

| Allocation example | Approximate size |
| --- | --- |
| One 640×360 BGR frame | 0.659 MiB |
| 16 such frames in the pool | 10.55 MiB |
| One 640×640×3 FP32 tensor row | 4.6875 MiB |
| Eight FP32 input rows, host | 37.5 MiB |
| Matching device input allocation | Another 37.5 MiB in this explicit-buffer design |
| Eight FP16 input rows, if input dtype is FP16 | 18.75 MiB per host/device buffer |

These figures exclude engine weights, execution-context activation memory, output buffers, decoder surfaces, GStreamer queues, snapshots, Python objects, camera handoffs, operating system, and other services. Frame references can be shared across objects, so adding every reference as a full allocation overestimates some stages. Engine workspace build settings are also not a total-runtime-memory limit.

## Capacity arithmetic without guessing performance

If eight cameras request 3 FPS each, arrival load is `8×3=24 images/s`. If measured full-batch service time is 400 ms for eight images, ideal batch throughput is `8/0.4=20 images/s`; backlog or dropping is inevitable. If measured full-batch service time is 200 ms, the equivalent is 40 images/s, but partial batches, decode contention, temperature, and other work can lower actual delivered capacity.

Measure both single/partial batches and the real camera mix. A throughput benchmark using synthetic tensors does not include the deployed capture pipeline. Preserve headroom rather than choosing sample rate equal to an optimistic maximum.
