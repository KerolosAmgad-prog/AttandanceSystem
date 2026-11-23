import tkinter as tk
import threading
import time
import cv2
from PIL import Image, ImageTk

from camera.webcam import Webcam
from recognition.face_engine import FaceEngine
from recognition.matcher import FaceMatcher
import config


class AttendanceApp:

    def __init__(self, root):

        self.root = root
        self.camera = None
        self.photo = None

        # Face Recognition components
        self.matcher = None
        self.face_engine = None

        # Threading and latest recognition state
        self.latest_results = []
        self.lock = threading.Lock()
        self.running = True
        self.recognition_thread = None

        # Debounce / display timer for welcome panel
        self.last_seen_time = 0
        self.active_person_name = None

        self.build_interface()

        # Initialize AI models
        self.init_ai_engine()

        # Start webcam and recognition worker
        self.start_camera()

        # Close event
        self.root.protocol(
            "WM_DELETE_WINDOW",
            self.on_close
        )

    def init_ai_engine(self):
        """Initializes the FaceMatcher and FaceEngine."""
        try:
            self.status_label.config(text="LOADING AI MODELS...", fg="#F59E0B")
            self.root.update_idletasks()

            # Load registered embeddings
            self.matcher = FaceMatcher(
                pickle_path=str(config.EMBEDDINGS_PATH),
                threshold=config.SIMILARITY_THRESHOLD
            )

            # Load buffalo_sc face detector & recognizer
            self.face_engine = FaceEngine(
                model_name=config.INSIGHTFACE_MODEL,
                det_size=config.DETECTION_SIZE
            )

            self.status_label.config(text="SYSTEM READY", fg="#16A34A")

        except Exception as error:
            print(f"[Error] Failed to initialize AI models: {error}")
            self.status_label.config(
                text=f"AI MODEL ERROR:\n{error}",
                fg="#EF4444"
            )

    def build_interface(self):

        self.root.configure(bg="#101820")

        # Header
        header = tk.Label(
            self.root,
            text="FACE RECOGNITION ATTENDANCE",
            font=("Arial", 19, "bold"),
            bg="#101820",
            fg="white"
        )
        header.pack(pady=8)

        # Main content
        self.content = tk.Frame(
            self.root,
            bg="#101820"
        )
        self.content.pack(
            fill="both",
            expand=True,
            padx=10,
            pady=10
        )

        # 60/40 layout
        self.content.grid_columnconfigure(0, weight=3)
        self.content.grid_columnconfigure(1, weight=2)
        self.content.grid_rowconfigure(0, weight=1)

        # Camera panel
        self.camera_frame = tk.Frame(
            self.content,
            bg="#1E293B"
        )
        self.camera_frame.grid(
            row=0,
            column=0,
            sticky="nsew",
            padx=5
        )

        self.camera_view = tk.Label(
            self.camera_frame,
            text="Waiting for camera...",
            bg="#1E293B",
            fg="white",
            font=("Arial", 18)
        )
        self.camera_view.pack(
            fill="both",
            expand=True
        )

        # Welcome panel
        self.welcome_frame = tk.Frame(
            self.content,
            bg="#F1F5F9"
        )
        self.welcome_frame.grid(
            row=0,
            column=1,
            sticky="nsew",
            padx=5
        )

        tk.Label(
            self.welcome_frame,
            text="Welcome!",
            font=("Arial", 25, "bold"),
            bg="#F1F5F9",
            fg="#0F172A"
        ).pack(pady=(75, 20))

        self.name_label = tk.Label(
            self.welcome_frame,
            text="Waiting for recognition...",
            font=("Arial", 17),
            bg="#F1F5F9",
            fg="#334155",
            wraplength=280
        )
        self.name_label.pack(pady=15)

        self.status_label = tk.Label(
            self.welcome_frame,
            text="STARTING UP...",
            font=("Arial", 14, "bold"),
            bg="#F1F5F9",
            fg="#16A34A"
        )
        self.status_label.pack(pady=20)

    def start_camera(self):

        try:
            self.camera = Webcam(
                device=config.CAMERA_DEVICE,
                width=config.CAMERA_WIDTH,
                height=config.CAMERA_HEIGHT,
                fps=config.CAMERA_FPS
            )
            self.camera.start()

            # Start background AI recognition worker thread
            self.recognition_thread = threading.Thread(
                target=self._recognition_loop,
                daemon=True
            )
            self.recognition_thread.start()

            # Start GUI frame update loop
            self.update_camera()

        except Exception as error:
            self.camera_view.config(
                text=f"Camera error:\n{error}"
            )
            self.status_label.config(
                text="CAMERA ERROR",
                fg="red"
            )

    def _recognition_loop(self):
        """
        Background worker that continuously processes the latest frame
        with InsightFace buffalo_sc and matches against embeddings.pkl.
        Keeps the GUI rendering thread running at a smooth 30 FPS.
        """
        while self.running:
            if self.camera is None or self.face_engine is None or self.matcher is None:
                time.sleep(0.05)
                continue

            frame = self.camera.get_frame()
            if frame is None:
                time.sleep(0.02)
                continue

            try:
                # Run detection + feature matching
                results = self.face_engine.process_and_match(frame, self.matcher)

                # Thread-safe update of latest results
                with self.lock:
                    self.latest_results = results

            except Exception as e:
                print(f"[Worker Error] Face processing exception: {e}")

            # Sleep slightly to prevent 100% CPU starvation
            time.sleep(0.03)

    def update_camera(self):

        if not self.running or self.camera is None:
            return

        frame = self.camera.get_frame()

        if frame is not None:
            # Read latest recognition results thread-safely
            with self.lock:
                current_results = list(self.latest_results)

            # Draw bounding boxes and names on the frame
            if self.face_engine is not None and current_results:
                display_frame = self.face_engine.draw_results(frame, current_results)
            else:
                display_frame = frame

            # Update Welcome / Status panel based on detection results
            self._update_panel_status(current_results)

            # OpenCV BGR -> RGB
            rgb_frame = cv2.cvtColor(
                display_frame,
                cv2.COLOR_BGR2RGB
            )

            # Calculate display dimensions
            panel_width = max(1, self.camera_frame.winfo_width() - 20)
            panel_height = max(1, self.camera_frame.winfo_height() - 20)

            image = Image.fromarray(rgb_frame)

            # Preserve aspect ratio
            image.thumbnail(
                (panel_width, panel_height),
                Image.Resampling.LANCZOS
            )

            # Convert to Tkinter image
            self.photo = ImageTk.PhotoImage(image)

            self.camera_view.config(
                image=self.photo,
                text=""
            )

        # Schedule next GUI update (approx 30 FPS)
        self.root.after(30, self.update_camera)

    def _update_panel_status(self, results):
        """Updates the name label and status text according to current detections."""
        now = time.time()

        if results:
            self.last_seen_time = now

            # Prioritize recognized known faces
            known_faces = [r for r in results if r["is_known"]]

            if known_faces:
                # Pick the match with highest score
                best_match = max(known_faces, key=lambda x: x["score"])
                name = best_match["name"]
                score = best_match["score"]

                self.active_person_name = name
                self.name_label.config(
                    text=f"{name}",
                    fg="#0F172A",
                    font=("Arial", 22, "bold")
                )
                self.status_label.config(
                    text=f"RECOGNIZED ({score:.0%})",
                    fg="#16A34A"
                )
            else:
                # Only unknown faces detected
                self.name_label.config(
                    text="Unknown Person",
                    fg="#DC2626",
                    font=("Arial", 18, "bold")
                )
                self.status_label.config(
                    text="NOT REGISTERED",
                    fg="#DC2626"
                )

        else:
            # If no face detected for over 2.5 seconds, reset panel
            if now - self.last_seen_time > 2.5:
                self.name_label.config(
                    text="Waiting for recognition...",
                    fg="#334155",
                    font=("Arial", 17)
                )
                self.status_label.config(
                    text="SYSTEM READY",
                    fg="#16A34A"
                )

    def on_close(self):
        self.running = False

        if self.camera is not None:
            self.camera.stop()

        self.root.destroy()
