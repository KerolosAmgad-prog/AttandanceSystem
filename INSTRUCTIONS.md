# Face Recognition Attendance System
## Raspberry Pi 4 Deployment & Operation Guide

This guide provides step-by-step instructions for deploying and running this Face Recognition Attendance System on a **Raspberry Pi 4 Model B** equipped with a standard **USB Webcam**.

---

## 1. System Requirements

* **Board**: Raspberry Pi 4 Model B (4GB or 8GB RAM recommended; 2GB supported with swap).
* **Operating System**: **Raspberry Pi OS (64-bit)** (Debian Bookworm or Bullseye).  
  > ⚠️ **Important**: 64-bit OS (`aarch64`) is required for modern `onnxruntime` and `insightface` pre-built wheels.
* **Camera**: Any UVC-compliant USB Webcam (e.g. Logitech C270, C920, or generic USB 2.0/3.0 webcam).
* **Power Supply**: Official 5V 3A USB-C power supply.
* **Cooling**: Passive heat sink or active cooling fan recommended for prolonged AI inference.

---

## 2. Prepare Raspberry Pi OS

Open the terminal on your Raspberry Pi (or connect via SSH):

### 2.1 Update System Packages
```bash
sudo apt update && sudo apt upgrade -y
```

### 2.2 Install Required System Libraries
Install Python development headers, Tkinter, OpenCV dependencies, and video tools:
```bash
sudo apt install -y python3-pip python3-venv python3-tk libgl1 libglib2.0-0 v4l-utils libatlas-base-dev
```

### 2.3 Grant Webcam Permissions
Ensure your user has access to the `/dev/video*` devices:
```bash
sudo usermod -aG video $USER
```
*(Log out and log back in, or reboot for group permissions to take effect).*

### 2.4 (Recommended) Increase Swap Memory
To prevent memory spikes during model loading:
```bash
sudo dphys-swapfile swapoff
sudo sed -i 's/^CONF_SWAPSIZE=.*/CONF_SWAPSIZE=2048/' /etc/dphys-swapfile
sudo dphys-swapfile setup
sudo dphys-swapfile swapon
```

---

## 3. Verify Webcam Connection

Plug your USB webcam into one of the blue **USB 3.0** ports on the Raspberry Pi 4.

Verify that the system detects the camera:
```bash
v4l2-ctl --list-devices
```
You should see output similar to:
```text
USB Camera (usb-0000:01:00.0-1.1):
    /dev/video0
    /dev/video1
```
*Note the primary device index (usually `0` for `/dev/video0`). If it is different, update `CAMERA_DEVICE` in `config.py`.*

---

## 4. Transfer Project Files to the Raspberry Pi

You can clone from Git or transfer the project folder from your PC using `scp` or a USB drive:

```bash
# Example using SCP from your Windows PC terminal:
scp -r "F:\Kerolos Amgad potfilo\AttandanceSystem" pi@<RASPBERRY_PI_IP>:~/AttandanceSystem
```

Navigate to the project folder on the Pi:
```bash
cd ~/AttandanceSystem
```

Ensure the following files are present:
- `models/embeddings.pkl` (your trained face embeddings database)
- `config.py`
- `main.py`
- `gui/app.py`
- `recognition/face_engine.py`
- `recognition/matcher.py`
- `camera/webcam.py`

---

## 5. Offline Model Transfer (Saves Download Time on Pi)

The system uses the `buffalo_sc` model. You can copy the already downloaded model from your PC to the Raspberry Pi:

On your Raspberry Pi, create the target folder:
```bash
mkdir -p ~/.insightface/models/buffalo_sc
```

Transfer the model files (`det_500m.onnx` and `w600k_mbf.onnx`) from your PC:
```bash
# On your Windows PC terminal:
scp "C:\Users\kerol\.insightface\models\buffalo_sc\*" pi@<RASPBERRY_PI_IP>:~/.insightface/models/buffalo_sc/
```

*(If you skip this step, InsightFace will automatically attempt to download the model on the first run, which requires internet access).*

---

## 6. Python Environment Setup

### 6.1 Create & Activate Virtual Environment
```bash
cd ~/AttandanceSystem
python3 -m venv .venv
source .venv/bin/activate
```

### 6.2 Upgrade Pip & Install Dependencies
```bash
pip install -U pip setuptools wheel
pip install -r requirements.txt
```

If you encounter issues installing `onnxruntime` on ARM64, install directly via:
```bash
pip install onnxruntime
pip install insightface
```

---

## 7. Configuration & Optimization

Open `config.py` to confirm settings:
```bash
nano config.py
```

* **`CAMERA_DEVICE = 0`**: Match the index from Step 3.
* **`DETECTION_SIZE = (320, 320)`**: Optimized for Raspberry Pi 4 CPU to achieve 15–20 FPS detection.
* **`SIMILARITY_THRESHOLD = 0.45`**: Cosine similarity cutoff. Adjust between `0.40` and `0.50` depending on your lighting conditions.

Save with `Ctrl + O`, then exit with `Ctrl + X`.

---

## 8. Testing & Operation

### 8.1 Test Camera Feed
Verify that the webcam opens cleanly using V4L2 backend:
```bash
python test_cam.py
```
*(Press `q` to exit).*

### 8.2 Test Face Detection & Recognition Pipeline
Verify model loading, embedding matching, and inference on the Pi's CPU:
```bash
python test_recognition_pipeline.py
```
Expected output:
```text
Loaded 30 registered faces
[FaceEngine] Initializing FaceAnalysis model 'buffalo_sc'...
[FaceEngine] Model 'buffalo_sc' ready with det_size=(320, 320)
ALL TESTS PASSED SUCCESSFULLY!
```

### 8.3 Launch the Full Attendance System
```bash
python main.py
```
* The application will launch in fullscreen.
* When a registered person steps in front of the camera, their name and similarity score appear on the video and on the Welcome panel.
* To exit the application, press the **ESC** key.

---

## 9. Auto-Start on Boot (Kiosk / Terminal Mode)

To make the attendance system start automatically when the Raspberry Pi powers on:

1. Create an autostart directory:
   ```bash
   mkdir -p ~/.config/autostart
   ```

2. Create a desktop entry:
   ```bash
   nano ~/.config/autostart/attendance.desktop
   ```

3. Paste the following configuration:
   ```ini
   [Desktop Entry]
   Type=Application
   Name=Attendance System
   Exec=/bin/bash -c "source /home/pi/AttandanceSystem/.venv/bin/activate && python /home/pi/AttandanceSystem/main.py"
   WorkingDirectory=/home/pi/AttandanceSystem
   Hidden=false
   NoDisplay=false
   X-GNOME-Autostart-enabled=true
   ```
   *(Replace `/home/pi/` with your actual home directory if your username is not `pi`).*

4. Save and reboot to test:
   ```bash
   sudo reboot
   ```

---

## 10. Troubleshooting

| Issue | Cause | Solution |
| :--- | :--- | :--- |
| `Cannot open webcam` | Incorrect camera index or missing permissions | Run `v4l2-ctl --list-devices` and update `CAMERA_DEVICE` in `config.py`. Run `sudo usermod -aG video $USER`. |
| `no display name and no $DISPLAY environment variable` | Running via headless SSH | Start GUI on the local screen: run `export DISPLAY=:0` before `python main.py`, or use VNC. |
| Slow FPS / High latency | CPU thermal throttling or oversized detection box | Ensure active fan cooling is running. Keep `DETECTION_SIZE = (320, 320)` in `config.py`. |
| `Illegal instruction` or ONNX error | 32-bit OS installed | Install **Raspberry Pi OS 64-bit** (`aarch64`). Check with `uname -m` (must return `aarch64`). |
