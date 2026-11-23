# Smart Face Recognition Attendance System

An automated, real-time face recognition attendance application designed for edge devices (Raspberry Pi 4, Linux) and desktop workstations (Windows). Built with **InsightFace (`buffalo_sc`)**, **OpenCV**, and **Tkinter**, the system achieves low-latency live video streaming alongside high-accuracy face identification against a pre-registered embedding database.

---

## Table of Contents
1. [System Architecture & Workflow](#system-architecture--workflow)
2. [Deep Learning Models Used & Rationale](#deep-learning-models-used--rationale)
3. [End-to-End Pipeline Steps](#end-to-end-pipeline-steps)
4. [Project Structure](#project-structure)
5. [Installation & Setup](#installation--setup)
6. [Configuration Guide](#configuration-guide)
7. [Running the Application](#running-the-application)
8. [Cross-Platform Support (PC & Raspberry Pi 4)](#cross-platform-support-pc--raspberry-pi-4)

---

## System Architecture & Workflow

```
 +-------------------------------------------------------------------+
 |                         Live USB / Webcam                         |
 +-------------------------------------------------------------------+
                                   │
                                   ▼
 +-------------------------------------------------------------------+
 |                   Webcam Thread (`webcam.py`)                     |
 |        - Background frame grabber with 1-frame queue              |
 |        - Zero-latency buffer drop; V4L2 / DirectShow backends     |
 +-------------------------------------------------------------------+
                                   │
                                   ▼
 +-------------------------------------------------------------------+
 |                AI Inference Worker (`app.py` Thread)              |
 |                                                                   |
 |  1. Detection & Alignment (SCRFD / `det_500m.onnx`)               |
 |     - Locates face bounding box and 5 facial landmarks            |
 |                                                                   |
 |  2. Feature Extraction (MobileFaceNet / `w600k_mbf.onnx`)         |
 |     - Extracts 512-dimensional normalized embedding vector        |
 |                                                                   |
 |  3. Cosine Matching (`matcher.py`)                                |
 |     - Matrix multiplication against `models/embeddings.pkl`       |
 |     - Threshold decision: Known (Score >= 0.45) vs Unknown        |
 +-------------------------------------------------------------------+
                                   │
                                   ▼
 +-------------------------------------------------------------------+
 |                     Tkinter GUI Main Thread                       |
 |  - 30 FPS non-blocking render loop                                |
 |  - Bounding box overlay & corner brackets on camera feed          |
 |  - Dynamic Welcome Panel with person's name and confidence %      |
 +-------------------------------------------------------------------+
```

---

## Deep Learning Models Used & Rationale

This project utilizes the **InsightFace `buffalo_sc`** (Small/Compact) model pack. It comprises two synergistic ONNX neural networks:

### 1. Face Detector: `det_500m.onnx` (SCRFD-500M)
* **Architecture**: Sample and Computation Redistribution for Efficient Face Detection (SCRFD).
* **Model Size**: ~2.5 MB.
* **Output**: Bounding boxes `[x1, y1, x2, y2]`, detection confidence score, and 5 facial landmark keypoints (left eye, right eye, nose tip, left mouth corner, right mouth corner).

### 2. Feature Extractor: `w600k_mbf.onnx` (MobileFaceNet)
* **Architecture**: MobileFaceNet trained with ArcFace loss on the extensive WebFace600K dataset.
* **Model Size**: ~13.6 MB.
* **Output**: A dense, discriminative feature vector of **512 floating-point values** ($\mathbb{R}^{512}$).

---

### Why We Use `buffalo_sc` (Design Rationale)

1. **Optimized for CPU & Edge Inference**:
   * Heavy models (such as `buffalo_l` / ResNet-100 or VGG-Face) require 250–600 ms per frame on CPUs and cause severe lag on single-board computers like the Raspberry Pi 4.
   * `buffalo_sc` completes detection, alignment, and feature extraction in **~30–50 ms on CPU**, enabling real-time operation without requiring dedicated GPU hardware.

2. **ArcFace Loss Discrimination Power**:
   * Trained with Additive Angular Margin Loss (ArcFace), which enforces both intra-class compactness and inter-class discrepancy on a hypersphere.
   * Even with its small footprint, it maintains high face-verification accuracy across varying poses, lighting conditions, and expressions.

3. **Ultra-Low Memory Footprint**:
   * Total model size is under **16 MB**, making it resilient against Out-Of-Memory (OOM) errors in resource-constrained environments.

4. **Integrated Alignment**:
   * The 5 landmark keypoints allow the engine to geometrically align and crop the face using affine transformation before feeding it into the recognition network, significantly improving accuracy.

---

## End-to-End Pipeline Steps

### Step 1: Threaded Video Acquisition
* Handled by `camera/webcam.py`.
* Operates in an independent daemon thread that continuously captures frames and buffers only the newest frame in a 1-item queue (`queue.Queue(maxsize=1)`).
* **Why?** Prevents internal OpenCV frame buffering delays, ensuring the AI model always operates on the most current real-time frame.

### Step 2: Face Detection & Landmark Extraction
* Handled by `recognition/face_engine.py` using `det_500m.onnx`.
* Downsamples or standardizes input frame to `(320, 320)` resolution for fast localization.
* Identifies all human faces in the scene and extracts 5 facial landmarks per face.

### Step 3: Facial Alignment & Embedding Generation
* The detected face chip is cropped and normalized using affine transformation based on landmark positions.
* The aligned crop is passed through `w600k_mbf.onnx` to generate a 512-dimensional embedding vector:
  $$\mathbf{e} \in \mathbb{R}^{512}$$
* The embedding is $L_2$-normalized:
  $$\mathbf{\hat{e}} = \frac{\mathbf{e}}{\|\mathbf{e}\|_2}$$

### Step 4: Vectorized Cosine Similarity Matching
* Handled by `recognition/matcher.py`.
* All registered reference embeddings stored in `models/embeddings.pkl` are normalized into matrix $\mathbf{E}_{\text{db}} \in \mathbb{R}^{N \times 512}$.
* Live vector matching is computed via matrix-vector dot product:
  $$\mathbf{s} = \mathbf{E}_{\text{db}} \cdot \mathbf{\hat{e}}_{\text{live}}$$
* Finds the best match index:
  $$i^* = \arg\max(\mathbf{s}), \quad s_{\text{best}} = \mathbf{s}[i^*]$$
* **Decision Rule**:
  * If $s_{\text{best}} \ge \text{threshold}$ (default `0.45`): Classify as registered person `names[i*]`.
  * If $s_{\text{best}} < \text{threshold}$: Classify as `"Unknown Person"`.

### Step 5: Decoupled Multi-Threaded GUI Rendering
* Handled by `gui/app.py`.
* **Inference Thread**: Continuously runs detection and matching, updating shared state thread-safely.
* **Tkinter Main Thread**: Refreshes at 30 FPS (`after(30, ...)`), drawing:
  * Green bounding boxes and badges for recognized members.
  * Red bounding boxes and warning tags for unregistered visitors.
  * Welcome sidebar showing the person's name and confidence percentage with a 2.5-second persistence debounce timer.

---

## Project Structure

```
AttandanceSystem/
├── config.py                      # Centralized configuration (paths, camera, thresholds)
├── main.py                        # Application entry point (fullscreen Tkinter window)
├── requirements.txt               # PIP package requirements
├── INSTRUCTIONS.md                # Raspberry Pi 4 deployment & kiosk autostart guide
├── README.md                      # Comprehensive project documentation
├── test_cam.py                    # Webcam standalone diagnostic tool
├── test_pkl.py                    # Embeddings database inspection script
├── test_recognition_pipeline.py   # Automated end-to-end AI pipeline verification
│
├── camera/
│   └── webcam.py                  # Threaded video capture with auto backend detection
│
├── gui/
│   └── app.py                     # Tkinter UI with decoupled AI worker thread
│
├── models/
│   └── embeddings.pkl             # Pre-trained face embeddings and identity labels
│
└── recognition/
    ├── face_engine.py             # InsightFace detector & visual drawing utilities
    └── matcher.py                 # Cosine similarity vector matcher
```

---

## Installation & Setup

### 1. Prerequisites
* Python 3.10, 3.11, or 3.12 (64-bit).
* Webcam connected via USB or built-in.

### 2. Clone / Open Repository
```bash
cd AttandanceSystem
```

### 3. Create Virtual Environment
* **On Windows (PowerShell)**:
  ```powershell
  python -m venv .venv
  .venv\Scripts\Activate.ps1
  ```
* **On Linux / Raspberry Pi**:
  ```bash
  python3 -m venv .venv
  source .venv/bin/activate
  ```

### 4. Install Dependencies
```bash
pip install -U pip setuptools wheel
pip install -r requirements.txt
```

*(Note: On Windows, you can also use `uv`: `uv sync` or `uv add insightface onnxruntime`)*.

---

## Configuration Guide

All primary options are located in `config.py`:

```python
# Camera settings
CAMERA_DEVICE = 0          # 0 for default webcam, or /dev/video0 on Linux
CAMERA_WIDTH = 640         # Frame width
CAMERA_HEIGHT = 480        # Frame height
CAMERA_FPS = 30            # Frame rate

# Model configuration
INSIGHTFACE_MODEL = "buffalo_sc"   # Model pack name
DETECTION_SIZE = (320, 320)        # Input resolution for face detector (320x320 recommended for CPU)
SIMILARITY_THRESHOLD = 0.45        # Cosine similarity threshold (0.40 - 0.50 recommended)
```

### Similarity Threshold Tuning
* **0.40**: More lenient, higher true positive rate under poor lighting, slightly higher risk of false acceptance.
* **0.45** *(Default)*: Balanced operating point for ArcFace MobileFaceNet embeddings.
* **0.50**: Strict security, minimal false acceptance rate, requires good lighting and frontal face orientation.

---

## Running the Application

### 1. Verify Pipeline (Diagnostic Test)
Run the automated test suite to ensure models and embeddings load properly:
```bash
python test_recognition_pipeline.py
```

### 2. Start Attendance Application
```bash
python main.py
```
* The application opens in fullscreen mode.
* Step in front of the camera: your face will be framed with a green box and your name will appear on the Welcome panel.
* To exit, press the **ESC** key.

---

## Cross-Platform Support (PC & Raspberry Pi 4)

The project includes built-in platform detection:
* **Windows**: Uses OpenCV `CAP_DSHOW` (DirectShow) backend for instant camera initialization.
* **Linux / Raspberry Pi OS**: Uses `CAP_V4L2` (Video4Linux2) backend.

For detailed hardware setup, swap memory tuning, and autostart on boot for Raspberry Pi 4, refer to **[INSTRUCTIONS.md](INSTRUCTIONS.md)**.
