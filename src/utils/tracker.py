# D:\Projects\OCR_01\src\utils\tracker.py
# Lightweight IoU tracker with debounce + sticky plate memory + helpers for line-crossing

from typing import List, Dict, Tuple

def iou_xyxy(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    iw = max(0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / max(ua, 1e-9)

class TrackManager:
    """
    - Greedy IoU matching
    - Debounce: limits how often a track 'fires' (emits an event) using frame-based timing
    - Sticky plate memory: once a non-empty OCR is seen, it persists for the track
    - Stores prev_center_y for virtual line direction checks (used by caller)
    """
    def __init__(
        self,
        enabled: bool = True,
        debounce_seconds: float = 3.0,
        max_age: int = 30,
        min_iou: float = 0.2
    ):
        self.enabled = bool(enabled)
        self.debounce_seconds = float(debounce_seconds)
        self.max_age = int(max_age)     # frames w/o match before deletion
        self.min_iou = float(min_iou)

        # tracks: dict[track_id] = {...}
        # fields: id, xyxy, plate_text, sticky_plate, age, last_seen_frame
        self.tracks: Dict[int, Dict] = {}
        self.next_id: int = 1

        # debounce memory: dict[track_id] -> last_fire_frame
        self.last_fire_frame: Dict[int, int] = {}

        # for virtual-line direction logic (caller uses it)
        # dict[track_id] -> last center y
        self.prev_center_y: Dict[int, int] = {}

        # FPS-aware debounce (set by caller)
        self.fps = 30.0  # default; caller should set with set_fps()
        self._update_debounce_frames()

        # internal frame index (optional)
        self.frame_index = 0

    def set_fps(self, fps: float):
        if fps and fps > 0:
            self.fps = float(fps)
        self._update_debounce_frames()

    def _update_debounce_frames(self):
        # how many frames must pass before the same track can 'fire' again
        self.debounce_frames = max(1, int(self.debounce_seconds * self.fps))

    def _assign(self, dets: List[List[int]]) -> List[Tuple[int, int]]:
        """
        Greedy IoU matching between current tracks and detections.
        Returns list of (track_id, det_index)
        """
        if not self.tracks or not dets:
            return []

        # Build IoU pairs
        pairs = []
        for tid, tr in self.tracks.items():
            for j, d in enumerate(dets):
                pairs.append((iou_xyxy(tr["xyxy"], d), tid, j))
        pairs.sort(reverse=True, key=lambda x: x[0])

        used_t, used_d = set(), set()
        matches = []
        for v, tid, j in pairs:
            if v < self.min_iou:
                break
            if tid in used_t or j in used_d:
                continue
            used_t.add(tid)
            used_d.add(j)
            matches.append((tid, j))
        return matches

    def update(self, rois: List[List[int]], texts: List[str], frame_idx: int) -> List[Dict]:
        """
        Inputs:
          - rois: list of [x1,y1,x2,y2] (license-plate boxes)
          - texts: list of OCR strings per roi (can be empty)
        Output:
          - list of track dicts: {id, xyxy, plate_text, fired(bool)}
            'plate_text' is sticky: once non-empty, it persists per track
        """
        self.frame_index = int(frame_idx)

        if not self.enabled:
            # one pseudo-track per detection (no persistence)
            return [
                {"id": i + 1, "xyxy": list(map(int, bb)),
                 "plate_text": (texts[i] if i < len(texts) else ""),
                 "fired": True}
                for i, bb in enumerate(rois)
            ]

        # 1) Age existing tracks
        for tid in list(self.tracks.keys()):
            self.tracks[tid]["age"] += 1

        # 2) Match detections to existing tracks
        matches = self._assign(rois)
        matched_dets = set(j for _, j in matches)

        # 3) Update matched tracks
        for tid, j in matches:
            bb = list(map(int, rois[j]))
            txt = (texts[j] if j < len(texts) else "") or ""
            tr = self.tracks[tid]

            tr["xyxy"] = bb
            tr["age"] = 0
            tr["last_seen_frame"] = self.frame_index

            # Sticky logic
            if txt:
                tr["plate_text"] = txt
                tr["sticky_plate"] = txt
            else:
                tr["plate_text"] = tr.get("sticky_plate", tr.get("plate_text", ""))

        # 4) Create new tracks for unmatched detections
        for j, bb in enumerate(rois):
            if j in matched_dets:
                continue
            tid = self.next_id
            self.next_id += 1
            txt = (texts[j] if j < len(texts) else "") or ""
            self.tracks[tid] = {
                "id": tid,
                "xyxy": list(map(int, bb)),
                "plate_text": txt,
                "sticky_plate": txt,
                "age": 0,
                "last_seen_frame": self.frame_index
            }

        # 5) Remove stale tracks
        for tid in list(self.tracks.keys()):
            if self.tracks[tid]["age"] > self.max_age:
                self.tracks.pop(tid, None)
                self.prev_center_y.pop(tid, None)
                self.last_fire_frame.pop(tid, None)

        # 6) Build output + debounce firing
        out = []
        for tid, tr in self.tracks.items():
            last_fire = self.last_fire_frame.get(tid, -10**9)
            fired = (self.frame_index - last_fire) >= self.debounce_frames
            if fired:
                self.last_fire_frame[tid] = self.frame_index

            disp_txt = tr.get("plate_text", "") or tr.get("sticky_plate", "")

            out.append({
                "id": tid,
                "xyxy": tr["xyxy"],
                "plate_text": disp_txt,
                "fired": fired
            })

        return out

    # If you really need a helper to call via the class, use a different name:
    @staticmethod
    def call_update(tracker: "TrackManager", rois: List[List[int]], texts: List[str], frame_idx: int) -> List[Dict]:
        return tracker.update(rois, texts, frame_idx)
