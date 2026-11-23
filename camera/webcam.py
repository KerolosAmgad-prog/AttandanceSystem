
import platform
import cv2
import threading
import queue


class Webcam:

    def __init__(
        self,
        device=0,
        width=640,
        height=480,
        fps=30
    ):

        # Cross-platform backend selection (Windows -> CAP_DSHOW, Linux/RPi -> CAP_V4L2)
        sys_name = platform.system()
        if sys_name == "Windows":
            backend = cv2.CAP_DSHOW
        elif sys_name == "Linux":
            backend = cv2.CAP_V4L2
        else:
            backend = cv2.CAP_ANY

        self.cap = cv2.VideoCapture(device, backend)

        if not self.cap.isOpened():
            raise RuntimeError("Cannot open webcam")

        # Camera configuration
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)

        # Store only the latest frame
        self.frame_queue = queue.Queue(maxsize=1)

        self.running = False
        self.thread = None

    def start(self):

        self.running = True

        self.thread = threading.Thread(
            target=self._capture_loop,
            daemon=True
        )

        self.thread.start()

    def _capture_loop(self):

        while self.running:

            success, frame = self.cap.read()

            if not success:
                continue

            # Remove an old frame if queue is full
            if self.frame_queue.full():

                try:
                    self.frame_queue.get_nowait()
                except queue.Empty:
                    pass

            # Insert latest frame
            try:
                self.frame_queue.put_nowait(frame)
            except queue.Full:
                pass

    def get_frame(self):

        try:
            return self.frame_queue.get_nowait()

        except queue.Empty:
            return None

    def stop(self):

        self.running = False

        if self.thread is not None:
            self.thread.join(timeout=2)

        self.cap.release()
