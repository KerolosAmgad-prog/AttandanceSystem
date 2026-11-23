import numpy as np
import cv2
from recognition.face_engine import FaceEngine
from recognition.matcher import FaceMatcher
import config

print("=" * 60)
print("TESTING FACE DETECTION AND RECOGNITION PIPELINE")
print("=" * 60)

# 1. Initialize Matcher
print("[1] Loading FaceMatcher with embeddings database...")
matcher = FaceMatcher(str(config.EMBEDDINGS_PATH), threshold=config.SIMILARITY_THRESHOLD)
print(f"    - Database contains {len(matcher.names)} registered entries")
print(f"    - Registered identities: {sorted(list(set(matcher.names)))}")

# 2. Initialize FaceEngine
print("\n[2] Initializing FaceEngine with model 'buffalo_sc'...")
engine = FaceEngine(
    model_name=config.INSIGHTFACE_MODEL,
    det_size=config.DETECTION_SIZE
)
print("    - FaceEngine successfully initialized.")

# 3. Test on Synthetic / Blank Image
print("\n[3] Testing detection on blank canvas (verifying no crash)...")
blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)
results = engine.process_and_match(blank_frame, matcher)
print(f"    - Blank frame test passed. Detected faces: {len(results)}")

# 4. Test drawing utility
print("\n[4] Testing draw_results utility...")
annotated = engine.draw_results(blank_frame, results)
assert annotated.shape == blank_frame.shape, "Annotated shape mismatch!"
print("    - draw_results executed cleanly.")

# 5. Verify compare with an actual registered embedding
print("\n[5] Verifying cosine similarity against registered database...")
first_embedding = matcher.embeddings[0]
name, score = matcher.compare(first_embedding)
print(f"    - Self-match test: Name='{name}', Cosine Similarity={score:.4f}")
assert name == matcher.names[0], "Failed self-match test!"
assert score > 0.99, "Self-match score should be ~1.0!"

print("\n" + "=" * 60)
print("ALL TESTS PASSED SUCCESSFULLY!")
print("=" * 60)
