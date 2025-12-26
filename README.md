Backend Local Setup (FastAPI + MySQL)
1) Requirements
Linux (Ubuntu/Debian recommended)


Python 3.10+


MySQL (local)


pip / venv


Optional system packages (helps with builds + OpenCV):
sudo apt-get update
sudo apt-get install -y build-essential python3-dev libgl1 libglib2.0-0


2) Install MySQL locally (script)
Run setup_mysql.sh in your backend root:
Cd Backend
chmod +x setup_mysql.sh
./setup_mysql.sh

Verify MySQL:
sudo systemctl status mysql --no-pager


4) Create venv + install requirements
From backend root (where requirements.txt exists):
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt


5) Run the backend
Run FastAPI:
python -m uvicorn main:app --host 0.0.0.0 --port 8080

Open:
API docs (Swagger): http://localhost:8080/docs



6) Quick API checks

Add a camera (example):
curl -X POST "http://localhost:8080/cameras" \
  -H "Content-Type: application/json" \
  -d '{"rtsp_url":"rtsp://user:pass@ip:554/stream","enabled":true}'

Stream preview (MJPEG in browser):
http://localhost:8000/cameras/<camera_uuid>/stream.mjpg?fps=10


Snapshot:
http://localhost:8000/cameras/<camera_uuid>/snapshot.jpg



Troubleshooting
MySQL not running
sudo systemctl restart mysql
sudo systemctl enable mysql

Access denied
sudo mysql -e "SELECT user, host FROM mysql.user;"

OpenCV import errors
sudo apt-get install -y libgl1 libglib2.0-0


