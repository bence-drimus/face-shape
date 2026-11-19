# /// script
# requires-python = ">=3.10,<3.13"
# dependencies = ["mediapipe==0.10.21"]
# ///
"""Estimate face shape and glasses sizes from a photo, fully on this machine.

Usage: uv run face_shape.py            (live webcam, press q to finish)
       uv run face_shape.py photo.jpg  (a photo instead)
       add --men or --women for one set of frame suggestions (default: both)
Face the camera straight on, hair off the forehead, neutral face, good light.

The only network call is a one-time download of Google's face landmark model
(no image is sent). After that it works with Wi-Fi off.
"""
import math
import sys
import urllib.request
from pathlib import Path

MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
MODEL = Path(__file__).with_name("face_landmarker.task")
IRIS_MM = 11.7  # average human iris diameter, used to turn pixels into mm

# Calibration knobs: rough guesses, nudge them if the result looks wrong for you.
LONG = 1.25  # length/cheek above this = long face (mesh stops below the hairline)
WIDE = 0.90  # forehead or jaw/cheek above this = "as wide as the cheekbones"
NARROW = 0.80  # below this = clearly narrower than the cheekbones


def classify(length, forehead, cheek, jaw):
    long_face = length / cheek > LONG
    f, j = forehead / cheek, jaw / cheek
    if j > WIDE:
        return "oblong" if long_face else "square"
    if f > WIDE and j < NARROW:
        return "heart"
    if f < NARROW and j < NARROW:
        return "diamond"
    return "oval" if long_face else "round"


FRAMES = {  # shape: (men, women)
    "oval": ("rectangle, square, pilot, oval", "most frames: oval, cat-eye, butterfly, rectangle"),
    "round": ("rectangle, square, browline", "rectangle, square, cat-eye"),
    "square": ("round, oval, phantos, thin rims", "round, oval, butterfly, thin rims"),
    "oblong": ("deep square, pilot, low bridge", "deep or oversized frames, low bridge"),
    "heart": ("rimless, pilot, light round", "rimless, light oval, bottom-heavy"),
    "diamond": ("browline, rectangle, oval", "cat-eye, browline, oval"),
}
WHO = {"--men": (0,), "--women": (1,)}.get(next((a for a in sys.argv if a in ("--men", "--women")), ""), (0, 1))
sys.argv = [a for a in sys.argv if a not in ("--men", "--women")]


def frames(shape):
    return " | ".join(("men: ", "women: ")[i] + FRAMES[shape][i] for i in WHO)


def landmarker():
    from mediapipe.tasks.python import vision
    from mediapipe.tasks.python.core.base_options import BaseOptions

    if not MODEL.exists():
        print("Downloading face landmark model (one time, no image sent)...")
        urllib.request.urlretrieve(MODEL_URL, MODEL)
    opts = vision.FaceLandmarkerOptions(base_options=BaseOptions(model_asset_path=str(MODEL)), num_faces=1)
    return vision.FaceLandmarker.create_from_options(opts)


def measure(lm, image):
    """Return pixel measurements and the mm-per-pixel scale, or None if no face."""
    result = lm.detect(image)
    if not result.face_landmarks:
        return None
    pts = result.face_landmarks[0]
    w, h = image.width, image.height

    def d(a, b):
        return math.dist((pts[a].x * w, pts[a].y * h), (pts[b].x * w, pts[b].y * h))

    iris_px = (d(469, 471) + d(474, 476)) / 2
    return {
        "length": d(10, 152),  # upper forehead to chin
        "forehead": d(103, 332),
        "cheek": d(234, 454),
        "jaw": d(172, 397),
        "temples": d(127, 356),
        "pd": d(468, 473),
        "mm": IRIS_MM / iris_px,
    }


def report(m):
    shape = classify(m["length"], m["forehead"], m["cheek"], m["jaw"])
    c = m["cheek"]
    return [
        f"length/cheek {m['length'] / c:.2f}  forehead/cheek {m['forehead'] / c:.2f}  jaw/cheek {m['jaw'] / c:.2f}",
        f"Face shape (rough): {shape} -> {frames(shape)}",
        f"Approx. pupillary distance: {m['pd'] * m['mm']:.0f} mm",
        frame_size(max(m["cheek"], m["temples"]) * m["mm"]),
    ]


def frame_size(width_mm):
    # ponytail: lens ~ (total - bridge 18 - two hinge pieces ~9 each) / 2; iris scale is +-5%
    lens = round((width_mm - 36) / 2)
    return f"Face width ~{width_mm:.0f} mm (+-{width_mm * 0.05:.0f}) -> frame total {width_mm - 5:.0f}-{width_mm + 5:.0f} mm, lens ~{lens - 1}-{lens + 1}, bridge 16-20"


def average(samples):
    return {k: sum(s[k] for s in samples) / len(samples) for k in samples[0]}


def live():
    import cv2
    import mediapipe as mp

    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        sys.exit("Can't open the camera. Allow camera access for your terminal in System Settings > Privacy & Security > Camera.")
    samples = []  # ponytail: last 30 frames averaged to steady the numbers
    with landmarker() as lm:
        while True:
            ok, frame = cam.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)  # mirror view
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            m = measure(lm, mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
            if m:
                samples = (samples + [m])[-30:]
            lines = report(average(samples)) if samples else ["Looking for a face..."]
            for i, line in enumerate(lines + ["q to finish"]):
                cv2.putText(frame, line, (10, 30 + 28 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.imshow("face shape (local only)", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    cam.release()
    cv2.destroyAllWindows()
    if samples:
        print("\n".join(report(average(samples))))


def check():
    assert classify(100, 80, 100, 95) == "square"
    assert classify(140, 80, 100, 95) == "oblong"
    assert classify(110, 95, 100, 70) == "heart"
    assert classify(110, 75, 100, 75) == "diamond"
    assert classify(140, 85, 100, 85) == "oval"
    assert classify(110, 85, 100, 85) == "round"
    assert "men: " in frames("oval") and "women: " in frames("oval")
    assert "lens ~51-53" in frame_size(140)  # 52-18 lenses on a 140 mm frame
    print("ok")


if __name__ == "__main__":
    if len(sys.argv) == 1:
        live()
    elif sys.argv[1] == "--check":
        check()
    else:
        import mediapipe as mp

        with landmarker() as lm:
            m = measure(lm, mp.Image.create_from_file(sys.argv[1]))
        if not m:
            sys.exit("No face found. Try a brighter, straight-on photo.")
        print("\n".join(report(m)))
