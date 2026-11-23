import cv2
import numpy as np
from insightface.app import FaceAnalysis


class FaceEngine:
    """
    Face detection and feature extraction engine using InsightFace (buffalo_sc).
    """

    def __init__(
        self,
        model_name="buffalo_sc",
        providers=None,
        det_size=(320, 320),
        ctx_id=0
    ):
        if providers is None:
            providers = ["CPUExecutionProvider"]

        self.model_name = model_name
        self.det_size = det_size

        print(f"[FaceEngine] Initializing FaceAnalysis model '{model_name}'...")
        self.app = FaceAnalysis(name=model_name, providers=providers)
        self.app.prepare(ctx_id=ctx_id, det_size=det_size)
        print(f"[FaceEngine] Model '{model_name}' ready with det_size={det_size}")

    def detect_faces(self, frame):
        """
        Runs face detection and alignment on an input BGR image.
        Returns a list of InsightFace Face objects.
        """
        if frame is None or frame.size == 0:
            return []

        # InsightFace expects BGR image (standard OpenCV format)
        return self.app.get(frame)

    def process_and_match(self, frame, matcher):
        """
        Detects faces in the frame and compares their 512-d embeddings
        against registered embeddings using FaceMatcher.

        Returns:
            list of dicts containing:
                - bbox: [x1, y1, x2, y2]
                - name: matched name (str) or None if unknown
                - score: float cosine similarity score
                - is_known: bool
        """
        faces = self.detect_faces(frame)
        results = []

        for face in faces:
            bbox = face.bbox.astype(int).tolist()
            name, score = matcher.compare(face.embedding)

            results.append({
                "bbox": bbox,
                "name": name,
                "score": float(score),
                "is_known": name is not None,
                "det_score": float(getattr(face, "det_score", 1.0))
            })

        return results

    @staticmethod
    def draw_results(frame, results):
        """
        Draws stylish bounding boxes and identification labels onto the frame.
        """
        annotated = frame.copy()

        for res in results:
            x1, y1, x2, y2 = res["bbox"]
            name = res["name"]
            score = res["score"]
            is_known = res["is_known"]

            # Visual styling: Green for recognized, Red for Unknown
            if is_known:
                box_color = (40, 180, 50)       # Green
                label_bg = (40, 180, 50)
                display_text = f"{name} ({score:.2f})"
            else:
                box_color = (40, 40, 220)       # Red
                label_bg = (40, 40, 220)
                display_text = f"Unknown ({score:.2f})" if score > 0 else "Unknown"

            # 1. Bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), box_color, 2)

            # 2. Corner accent markers for modern UI look
            corner_len = min(20, max(5, (x2 - x1) // 6))
            thick = 3
            # Top-left
            cv2.line(annotated, (x1, y1), (x1 + corner_len, y1), box_color, thick)
            cv2.line(annotated, (x1, y1), (x1, y1 + corner_len), box_color, thick)
            # Top-right
            cv2.line(annotated, (x2, y1), (x2 - corner_len, y1), box_color, thick)
            cv2.line(annotated, (x2, y1), (x2, y1 + corner_len), box_color, thick)
            # Bottom-left
            cv2.line(annotated, (x1, y2), (x1 + corner_len, y2), box_color, thick)
            cv2.line(annotated, (x1, y2), (x1, y2 - corner_len), box_color, thick)
            # Bottom-right
            cv2.line(annotated, (x2, y2), (x2 - corner_len, y2), box_color, thick)
            cv2.line(annotated, (x2, y2), (x2, y2 - corner_len), box_color, thick)

            # 3. Label tag above bounding box
            (text_w, text_h), baseline = cv2.getTextSize(
                display_text, cv2.FONT_HERSHEY_DUPLEX, 0.55, 1
            )
            label_y1 = max(0, y1 - text_h - 12)
            label_y2 = y1

            cv2.rectangle(
                annotated,
                (x1, label_y1),
                (x1 + text_w + 12, label_y2),
                label_bg,
                -1
            )
            cv2.putText(
                annotated,
                display_text,
                (x1 + 6, label_y2 - 5),
                cv2.FONT_HERSHEY_DUPLEX,
                0.55,
                (255, 255, 255),
                1,
                cv2.LINE_AA
            )

        return annotated
