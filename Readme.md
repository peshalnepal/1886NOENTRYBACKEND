# RTSP Multi-Camera Object Detection (Jetson Nano) — Setup Guide

This README helps anyone set up the **RTSP multi-camera object detection system** from scratch on an **NVIDIA Jetson Nano** with limited internal storage.

You maintain **two Python runtimes**:

* **Env A (Python 3.11)**: OpenCV **built from source with GStreamer enabled** (for reliable RTSP ingest via OpenCV + GStreamer).
* **Env B (Python 3.6)**: **TensorRT** runtime (JetPack-provided) used by the TensorRT inference service.

This guide also includes how to mount external storage (e.g., **64GB SD card / USB SSD**) as the primary location for:

* project code
* pyenv + Python builds
* virtual environments
* OpenCV sources/build artifacts
* TensorRT engines/models
* clips / logs

---

## Folder structure (current)

Your repo root looks like:


  * `Backend/`

    * `tensort/`

      * `models/`  (TensorRT engines are already here)
      * `trt_infer_service.py`  (run with Python 3.6)
    * `setup_mysql.sh`  (**creates/sets up internal PostgreSQL**)
    * `requirements.txt`
  * `Frontend/`

In this README, we assume you clone the repo into:

* `/data/projects/`

---

## 0) External storage layout (recommended)

Mount your external storage at:

* **`/data`**

Then keep big folders on `/data`:

* `/data/projects/` — git repos
* `/data/pyenv/` — pyenv + compiled Pythons
* `/data/venvs/` — virtual envs
* `/data/build/` — OpenCV sources/builds
* `/data/videos/` — clips/ring buffers
* `/data/logs/` — logs

This keeps the internal 16GB mostly OS-only.

---

## 1) Hardware & OS assumptions

* Jetson Nano with JetPack installed.
* You have `sudo` access.
* Internet access for apt/pip/git.

---

## 2) Add external storage and mount as `/data`

### 2.1 Identify the disk

Insert SD/USB, then:

```bash
lsblk -o NAME,SIZE,FSTYPE,MOUNTPOINT,LABEL
```

Find the new device (examples: `/dev/mmcblk1` for SD, `/dev/sda` for USB SSD).

### 2.2 Partition and format (WARNING: wipes the disk)

Example for SD card `/dev/mmcblk1`:

```bash
sudo parted /dev/mmcblk1 --script mklabel gpt
sudo parted /dev/mmcblk1 --script mkpart primary ext4 0% 100%
sudo mkfs.ext4 -L DATA /dev/mmcblk1p1
```

### 2.3 Mount to `/data`

```bash
sudo mkdir -p /data
sudo mount /dev/mmcblk1p1 /data
df -h /data
```

### 2.4 Persist mount in `/etc/fstab`

Get UUID:

```bash
sudo blkid /dev/mmcblk1p1
```

Edit fstab:

```bash
sudo nano /etc/fstab
```

Add a line (replace UUID):

```text
UUID=xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx  /data  ext4  defaults,noatime  0  2
```

Test:

```bash
sudo umount /data
sudo mount -a
df -h /data
```

### 2.5 Create folders on `/data`

```bash
sudo mkdir -p /data/{projects,pyenv,venvs,build,videos,logs}
sudo chown -R $USER:$USER /data
```

---

## 3) Install system prerequisites

Update and install base tooling:

```bash
sudo apt-get update
sudo apt-get install -y \
  git curl wget unzip ca-certificates \
  build-essential pkg-config \
  cmake ninja-build \
  ffmpeg \
  software-properties-common \
  libssl-dev zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev \
  libffi-dev liblzma-dev
```

---

## 4) Install pyenv to `/data` and build Python 3.11

### 4.1 Install pyenv

```bash
git clone https://github.com/pyenv/pyenv.git /data/pyenv/.pyenv
```

### 4.2 Add pyenv to your shell

Append to `~/.bashrc`:

```bash
export PYENV_ROOT="/data/pyenv/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init -)"
```

Reload:

```bash
source ~/.bashrc
pyenv --version
```

### 4.3 Install Python build dependencies

```bash
sudo apt-get install -y \
  make gcc g++ \
  libncursesw5-dev xz-utils tk-dev \
  libxml2-dev libxmlsec1-dev
```

### 4.4 Build Python 3.11

Example:

```bash
pyenv install 3.11.7
pyenv global 3.11.7
python --version
```

---

## 5) Env A (Python 3.11): Create venv named `venv`

```bash
python -m venv /data/venvs/venv
source /data/venvs/venv/bin/activate
python -m pip install --upgrade pip wheel setuptools
```

---

## 6) Build OpenCV from source with GStreamer (install into venv)

### 6.1 Install OpenCV + GStreamer deps

```bash
sudo apt-get install -y \
  libjpeg-dev libpng-dev libtiff-dev \
  libavcodec-dev libavformat-dev libswscale-dev libv4l-dev \
  libxvidcore-dev libx264-dev \
  libgtk-3-dev libatlas-base-dev gfortran \
  libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev \
  gstreamer1.0-tools gstreamer1.0-plugins-base \
  gstreamer1.0-plugins-good gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly \
  gstreamer1.0-libav
```

### 6.2 Download OpenCV sources

```bash
cd /data/build
git clone --branch 4.8.0 --depth 1 https://github.com/opencv/opencv.git
git clone --branch 4.8.0 --depth 1 https://github.com/opencv/opencv_contrib.git
```

### 6.3 Configure CMake for Python 3.11 venv + GStreamer

Activate venv:

```bash
source /data/venvs/venv/bin/activate
```

Compute Python paths:

```bash
PY_BIN=$(which python)
PY_INC=$($PY_BIN -c "import sysconfig; print(sysconfig.get_paths()['include'])")
PY_SITE=$($PY_BIN -c "import site; print(site.getsitepackages()[0])")
```

Configure build:

```bash
mkdir -p /data/build/opencv/build
cd /data/build/opencv/build

cmake -G Ninja \
  -D CMAKE_BUILD_TYPE=Release \
  -D CMAKE_INSTALL_PREFIX=/data/venvs/venv \
  -D OPENCV_EXTRA_MODULES_PATH=/data/build/opencv_contrib/modules \
  -D BUILD_opencv_python3=ON \
  -D PYTHON3_EXECUTABLE="$PY_BIN" \
  -D PYTHON3_INCLUDE_DIR="$PY_INC" \
  -D PYTHON3_PACKAGES_PATH="$PY_SITE" \
  -D OPENCV_PYTHON3_INSTALL_PATH="$PY_SITE" \
  -D WITH_GSTREAMER=ON \
  -D WITH_FFMPEG=ON \
  -D BUILD_TESTS=OFF -D BUILD_PERF_TESTS=OFF -D BUILD_EXAMPLES=OFF \
  /data/build/opencv
```

Build + install:

```bash
ninja -j$(nproc)
ninja install
```

### 6.4 Verify OpenCV + GStreamer

```bash
source /data/venvs/venv/bin/activate
python -c "import cv2; print(cv2.__version__)"
python -c "import cv2; print(cv2.getBuildInformation())" | grep -i gstreamer
```

Expected: `GStreamer: YES`

---

## 7) Clone the repo to `/data/projects`

```bash
cd /data/projects
git clone <YOUR_GIT_URL>
```

---

## 8) Backend database: create/setup internal PostgreSQL

Your `Backend/setup_mysql.sh` script creates/sets up **internal PostgreSQL**.

Run it from `Backend/`:

```bash
cd /data/projects/Backend
chmod +x setup_mysql.sh
sudo ./setup_mysql.sh
```

If the script prints connection info (user/password/dbname/port), keep it for your `.env` / config.

---

## 9) Install Backend dependencies (Env A: Python 3.11)

Activate Env A and install Backend requirements:

```bash
source /data/venvs/venv/bin/activate
cd /data/projects/Backend
pip install -r requirements.txt
```

Important:

* If `requirements.txt` includes `opencv-python` or `opencv-python-headless`, remove it to avoid overwriting your custom OpenCV build.

---

## 10) Env B (Python 3.6): TensorRT runtime for `Backend/tensort/trt_infer_service.py`

You want to run the TensorRT service like this:

```bash
cd /data/projects/Backend/tensort
python3.6 ./trt_infer_service.py
```

### What `trt_infer_service.py` needs

This service imports:

- `tensorrt`
- `pycuda.driver` + `pycuda.autoinit`
- `cv2`
- `numpy`
- `flask`

On Jetson Nano, **TensorRT and PyCUDA should come from JetPack** (system packages), not pip.

### 10.1 Verify Python 3.6 exists

```bash
python3.6 --version
```

If missing (only if supported by your OS repositories):

```bash
sudo apt-get install -y python3.6 python3.6-venv python3.6-dev
```

### 10.2 Verify JetPack-provided TensorRT + PyCUDA + OpenCV bindings

First, check if the key modules already work in **system Python 3.6**:

```bash
python3.6 -c "import tensorrt as trt; print('tensorrt', trt.__version__)"
python3.6 -c "import pycuda.driver as cuda; print('pycuda OK')"
python3.6 -c "import cv2; print('cv2', cv2.__version__)"
python3.6 -c "import numpy as np; print('numpy', np.__version__)"
```

If all of the above succeed, you can skip installing most packages and go to **10.4**.

### 10.3 Install missing system packages (only if the checks fail)

JetPack images often already include these. If something is missing, install via `apt`.

#### Useful checks

```bash
dpkg -l | grep -E "nvinfer|tensorrt|cuda|opencv" \
  | sed -e 's/^/  /'
```

#### Common packages (names can vary by JetPack/Ubuntu)

```bash
sudo apt-get update
sudo apt-get install -y \
  python3-pip python3-setuptools \
  python3-numpy \
  libnvinfer-dev \
  python3-libnvinfer
```

For PyCUDA, package availability varies. Try:

```bash
apt-cache search pycuda
```

If you see a `python3-pycuda` package, install it:

```bash
sudo apt-get install -y python3-pycuda
```

> Avoid installing `pycuda` from pip on Jetson unless you absolutely must. It often requires compiling and matching CUDA toolchains.

### 10.4 Create optional venv for Python 3.6 (recommended)

To keep the service’s pure-python dependencies isolated **while still using JetPack system bindings**, create a venv that inherits system packages:

```bash
python3.6 -m venv --system-site-packages /data/venvs/venvtensorrt
source /data/venvs/venvtensorrt/bin/activate
python -m pip install --upgrade pip wheel setuptools
```

Install only what is not guaranteed to be on JetPack (typically Flask):

```bash
pip install "flask>=1.1,<3.0"
```

Quick smoke test inside the venv:

```bash
python3.6 -c "import tensorrt, pycuda.driver as cuda, cv2, numpy, flask; print('TRT env OK')"
```

### 10.5 Engines/models path

Your engines are already present (no engine build required). By default, the service uses:

- `Backend/tensort/models/yolov8n.engine`
- `Backend/tensort/models/yolov8n-pose.engine`

Make sure you run the service from the `Backend/tensort/` directory so relative paths resolve:

```bash
cd /data/projects/Backend/tensort
ls -lah models/
```

---

## 11) Run services

### 11.1 Run Backend API (FastAPI) — Env A (Python 3.11)

```bash
source /data/venvs/venv/bin/activate
cd /data/projects/Backend
uvicorn main:app --host 0.0.0.0 --port 8080
```

Open:

* `http://<JETSON_IP>:8080/docs`

> If your entrypoint is not `main:app`, replace it (example: `app.main:app`).

### 11.2 Run TensorRT inference service — Env B (Python 3.6)

Your TensorRT engines are already present (no engine build needed). The engines/models are located at:

* `/data/projects/Backend/tensort/models/`

Run the TRT service like you asked:

```bash
cd /data/projects/Backend/tensort
python3.6 ./trt_infer_service.py
```

Recommended (ensures dependencies resolve consistently):

```bash
source /data/venvs/venvtensorrt/bin/activate
cd /data/projects/Backend/tensort
python3.6 ./trt_infer_service.py
```

---

## 12) Frontend (Vite) setup

From repo root:

```bash
cd /data/projects/Frontend
npm install
npm start
```

If your Frontend expects a backend base URL, set it in your frontend config (for example `Frontend/src/config/api.js`) to point to:

* `http://<JETSON_IP>:8080`

---

## 13) RTSP validation and troubleshooting

### 13.1 Validate RTSP with GStreamer

```bash
gst-launch-1.0 rtspsrc location="rtsp://USER:PASS@IP/..." latency=200 ! decodebin ! videoconvert ! autovideosink
```

### 13.2 Validate RTSP with OpenCV (Python 3.11 env)

```bash
source /data/venvs/venv/bin/activate
python - <<'PY'
import cv2
url="rtsp://USER:PASS@IP/..."
cap=cv2.VideoCapture(url)
print("opened:", cap.isOpened())
ok, frame = cap.read()
print("read:", ok, None if frame is None else frame.shape)
cap.release()
PY
```

### 13.3 If you see H264/FFmpeg warnings

Some streams show warnings but still work. If frames stall:

* reduce `sample_fps` per camera
* resize frames before inference
* prefer GStreamer decode on Jetson for stability

---

## 14) Make `/data` the “main” place (safe approach)

Fully moving the OS root filesystem is risky on Jetson. The safe approach is:

* Keep OS on internal 16GB
* Put **everything you control** on `/data`

Optional: redirect caches to `/data`:

Add to `~/.bashrc`:

```bash
export PIP_CACHE_DIR=/data/.cache/pip
export XDG_CACHE_HOME=/data/.cache
```

Create cache folders:

```bash
mkdir -p /data/.cache/pip
```

---

## 15) Checklist for a fresh setup

1. Mount external storage at `/data`
2. Install system deps
3. Install pyenv + Python 3.11
4. Create `/data/venvs/venv`
5. Build OpenCV with GStreamer into `venv`
6. Clone repo into `/data/projects`
7. Run `Backend/setup_mysql.sh` (PostgreSQL setup)
8. Install Backend requirements in `venv`
9. Setup Python 3.6 + create `/data/venvs/venvtensorrt`
10. Run TensorRT service with `python3.6 ./trt_infer_service.py`
11. Run Backend API with `uvicorn`
12. Start Frontend with `npm start`
13. Validate RTSP streams using GStreamer first
