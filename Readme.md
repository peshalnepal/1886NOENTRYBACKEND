# RTSP Multi-Camera Object Detection (Jetson Nano Backend + PC Frontend) — Setup Guide

This README helps anyone set up the **RTSP multi-camera object detection system** where:

* **Backend (FastAPI + RTSP ingest + detection runtime)** runs on the **Jetson Nano**.
* **Frontend (Vite UI)** runs on a **separate computer** on the **same network (LAN/Wi‑Fi)**.

Because the frontend is not running on the Jetson, **you must use the Jetson’s LAN IP address** (not `localhost`) when configuring the UI.

---

## Folder structure (current)

Your repo root looks like:

* `Backend/`

  * `tensort/`

    * `models/` (TensorRT engines are already here)
    * `trt_infer_service.py` (run with Python 3.6)
  * `setup_mysql.sh` (creates/sets up internal PostgreSQL)
  * `requirements.txt`
* `Frontend/`

In this README, we assume you clone the repo into:

* `/data/projects/`

---

## 0) Deployment layout (important)

### Backend (Jetson Nano)

Runs on Jetson:

* FastAPI backend (port **8080**) serving API + Swagger docs
* TensorRT inference service (Python 3.6) if used by your backend
* Database (internal PostgreSQL created by `setup_mysql.sh`)

### Frontend (Separate computer on same network)

Runs on your laptop/desktop:

* Vite dev server (example port **9000**) serving the UI
* UI calls the Jetson backend using the Jetson IP:

  * `http://<JETSON_IP>:8080`

---

## 1) External storage layout (recommended)

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

## 2) Find the Jetson Nano IP address (required for UI)

You will do this **on the Jetson**.

### 2.1 Show the Jetson LAN IP

Run:

```bash
hostname -I
```

You may see multiple IPs (Wi‑Fi + Ethernet). Choose the LAN IP that matches your network, commonly:

* `192.168.x.x` (home routers)
* `10.0.0.x` (some routers / office networks)

Alternative (shows interface + IP clearly):

```bash
ip -4 addr show
```

### 2.2 Confirm the backend is reachable from the frontend computer

From the **frontend computer**, test:

```bash
ping <JETSON_IP>
```

Then open Swagger docs in a browser:

* `http://<JETSON_IP>:8080/docs`

If this does not load:

* ensure Jetson and the computer are on the **same network**
* ensure the backend is started with `--host 0.0.0.0`
* check firewall rules (allow inbound TCP 8080)

---

## 3) Jetson setup steps (Backend)

You maintain **two Python runtimes**:

* **Env A (Python 3.11)**: OpenCV built from source with GStreamer enabled (for reliable RTSP ingest).
* **Env B (Python 3.6)**: JetPack TensorRT runtime (system bindings) used by `trt_infer_service.py`.

All steps below are performed on the **Jetson Nano**.

### 3.1 Install system prerequisites

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

### 3.2 Install pyenv to `/data` and build Python 3.11

```bash
git clone https://github.com/pyenv/pyenv.git /data/pyenv/.pyenv
```

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

Build deps:

```bash
sudo apt-get install -y \
  make gcc g++ \
  libncursesw5-dev xz-utils tk-dev \
  libxml2-dev libxmlsec1-dev
```

Install:

```bash
pyenv install 3.11.7
pyenv global 3.11.7
python --version
```

### 3.3 Env A (Python 3.11): Create venv named `venv`

```bash
python -m venv /data/venvs/venv
source /data/venvs/venv/bin/activate
python -m pip install --upgrade pip wheel setuptools
```

### 3.4 Build OpenCV from source with GStreamer (install into venv)

Install deps:

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

Download sources:

```bash
cd /data/build
git clone --branch 4.8.0 --depth 1 https://github.com/opencv/opencv.git
git clone --branch 4.8.0 --depth 1 https://github.com/opencv/opencv_contrib.git
```

Configure + build:

```bash
source /data/venvs/venv/bin/activate

PY_BIN=$(which python)
PY_INC=$($PY_BIN -c "import sysconfig; print(sysconfig.get_paths()['include'])")
PY_SITE=$($PY_BIN -c "import site; print(site.getsitepackages()[0])")

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

ninja -j$(nproc)
ninja install
```

Verify:

```bash
source /data/venvs/venv/bin/activate
python -c "import cv2; print(cv2.__version__)"
python -c "import cv2; print(cv2.getBuildInformation())" | grep -i gstreamer
```

Expected: `GStreamer: YES`

### 3.5 Clone the repo to `/data/projects`

```bash
cd /data/projects
git clone <YOUR_GIT_URL>
```

### 3.6 Backend database: create/setup internal PostgreSQL

```bash
cd /data/projects/Backend
chmod +x setup_mysql.sh
sudo ./setup_mysql.sh
```

### 3.7 Install Backend dependencies (Env A: Python 3.11)

```bash
source /data/venvs/venv/bin/activate
cd /data/projects/Backend
pip install -r requirements.txt
```

Important:

* If `requirements.txt` includes `opencv-python` or `opencv-python-headless`, remove it to avoid overwriting your custom OpenCV build.

---

## 4) TensorRT service (Jetson, Env B: Python 3.6)

Run the TensorRT service like this:

```bash
cd /data/projects/Backend/tensort
python3.6 ./trt_infer_service.py
```

Optional (recommended): create a venv that inherits JetPack system packages:

```bash
python3.6 -m venv --system-site-packages /data/venvs/venvtensorrt
source /data/venvs/venvtensorrt/bin/activate
python -m pip install --upgrade pip wheel setuptools
pip install "flask>=1.1,<3.0"

cd /data/projects/Backend/tensort
python3.6 ./trt_infer_service.py
```

---

## 5) Run the backend services on Jetson

### 5.1 Run Backend API (FastAPI) — Env A (Python 3.11)

```bash
source /data/venvs/venv/bin/activate
cd /data/projects/Backend
uvicorn main:app --host 0.0.0.0 --port 8080
```

Open Swagger docs from your **frontend computer**:

* `http://<JETSON_IP>:8080/docs`

> If your entrypoint is not `main:app`, replace it (example: `app.main:app`).

---

## 6) Frontend setup (runs on a separate computer)

All steps in this section happen on your **computer** (not the Jetson).

### 6.1 Install and run the UI

```bash
cd Frontend
npm install
npm start
```

Your Vite dev server will typically run on:

* `http://<YOUR_COMPUTER_IP>:9000`

### 6.2 Configure the UI to talk to the Jetson backend (store the Jetson IP)

The UI must use:

* `http://<JETSON_IP>:8080`

#### Option A (recommended for your current UI): Store the backend base URL in the UI settings

1. Open the frontend UI in the browser.

2. Find the **API Base URL** field.

3. Set it to:

   * `http://<JETSON_IP>:8080`

4. Save/apply.

This should store the value in the browser (local storage) so you do not need to re-enter it.

#### Option B: Use a Vite env file (if your frontend supports it)

Create `Frontend/.env.local`:

```bash
VITE_API_BASE_URL=http://<JETSON_IP>:8080
```

Restart the frontend:

```bash
npm start
```

### 6.3 Quick validation from the frontend computer

Open:

* `http://<JETSON_IP>:8080/docs`

If it works, the UI should also be able to load cameras, streams, and updates.

---

## 7) RTSP validation and troubleshooting

### 7.1 Validate RTSP with GStreamer (Jetson)

```bash
gst-launch-1.0 rtspsrc location="rtsp://USER:PASS@IP/..." latency=200 ! decodebin ! videoconvert ! autovideosink
```

### 7.2 Validate RTSP with OpenCV (Jetson, Python 3.11 env)

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

### 7.3 If you see H264/FFmpeg warnings

Some streams show warnings but still work. If frames stall:

* reduce `sample_fps` per camera
* resize frames before inference
* prefer GStreamer decode on Jetson for stability

---

## 8) Checklist for a fresh setup

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
11. Run Backend API with `uvicorn --host 0.0.0.0 --port 8080`
12. On your **computer**, start the Frontend with `npm start`
13. **Find the Jetson IP** and **store it** in the UI as `http://<JETSON_IP>:8080`
14. Validate Swagger: `http://<JETSON_IP>:8080/docs`
