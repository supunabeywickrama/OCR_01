import cv2

class VideoSink:
    def __init__(self, path, size, fps):
        self.path = path
        self.size = size
        self.fps = fps or 25

        # Try multiple codecs/extensions for Windows
        candidates = [
            ("mp4v", "mp4"),
            ("XVID", "avi"),
            ("MJPG", "avi")
        ]

        self.writer = None
        for fourcc_str, ext in candidates:
            fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
            out_path = path
            if not out_path.lower().endswith(f".{ext}"):
                out_path = path.rsplit(".", 1)[0] + f".{ext}"
            w = cv2.VideoWriter(out_path, fourcc, self.fps, self.size)
            if w.isOpened():
                self.writer = w
                self.path = out_path
                print(f"[INFO] VideoSink using {fourcc_str} → {self.path}")
                break

        if self.writer is None:
            raise RuntimeError("Failed to open VideoWriter with mp4v/XVID/MJPG")

    def write(self, frame):
        self.writer.write(frame)

    def release(self):
        self.writer.release()
