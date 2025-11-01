# D:\Projects\OCR_01\src\license_plate_blurring.py
# LP detect/blur + OCR + clarity scoring + enhancement + tracking/debounce
# + CSV/JSON + crops + overlays + analytics
# + Virtual line counting (Inbound/Outbound) + recent plates panel
# + Vehicle-wide coverage: every vehicle gets a plate attempt; else "UNREADABLE"

import argparse, os, cv2, csv, json, yaml, time, hashlib, collections
from datetime import datetime
import numpy as np
from ultralytics import YOLO
import supervision as sv

from .utils.blur import apply_blur_to_regions
from .utils.video import VideoSink
from .utils.ocr import OCR
from .utils.tracker import TrackManager

# ------------------------- helpers ------------------------- #
def load_config(path: str):
    print(f"[INFO] Loading config: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def parse_override(kv):
    k, v = kv.split("=", 1)
    for caster in (lambda x: json.loads(x), int, float):
        try:
            return k, caster(v)
        except Exception:
            pass
    return k, v

def set_in_dict(d, dotted_key, value):
    parts = dotted_key.split("."); cur = d
    for p in parts[:-1]:
        if p not in cur or not isinstance(cur[p], dict): cur[p] = {}
        cur = cur[p]
    cur[parts[-1]] = value

def clip(v, lo, hi): return max(lo, min(hi, v))

def expand_box(x1, y1, x2, y2, pad_ratio, W, H):
    w, h = x2 - x1, y2 - y1
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    side = int(max(w, h) * pad_ratio); half = side // 2
    nx1 = clip(cx - half, 0, W - 1); ny1 = clip(cy - half, 0, H - 1)
    nx2 = clip(cx + half, 0, W - 1); ny2 = clip(cy + half, 0, H - 1)
    return nx1, ny1, nx2, ny2

def clean_text(t, uppercase=True, alnum_only=True):
    if t is None: return ""
    if uppercase: t = t.upper()
    if alnum_only: t = "".join(ch for ch in t if ch.isalnum())
    return t

def bbox_iou(a,b):
    ax1,ay1,ax2,ay2=a; bx1,by1,bx2,by2=b
    iw=max(0,min(ax2,bx2)-max(ax1,bx1)); ih=max(0,min(ay2,by2)-max(ay1,by1))
    inter=iw*ih; ua=(ax2-ax1)*(ay2-ay1)+(bx2-bx1)*(by2-by1)-inter
    return inter/max(ua,1e-9)

# ---- clarity scoring + enhancement ----
def measure_sharpness(img):
    if img is None or img.size == 0: return 0.0
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())

def enhance_plate(img):
    if img is None or img.size == 0: return img
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    eq = clahe.apply(gray)
    blur = cv2.GaussianBlur(eq, (0,0), 3)
    unsharp = cv2.addWeighted(eq, 1.5, blur, -0.5, 0)
    return cv2.cvtColor(unsharp, cv2.COLOR_GRAY2BGR)

def resolve_line_points(vl_points, W, H, normalized=False, y_only=None):
    """
    Returns two pixel points [(x1,y1),(x2,y2)] in original frame space.
    - normalized=True: interpret points in [0..1] relative coords
    - y_only: if set (float or int), draw a full-width horizontal line at this y
    """
    if y_only is not None:
        y = int(round(y_only if not normalized else y_only * H))
        return [(0, y), (W-1, y)]
    if normalized:
        (x1, y1), (x2, y2) = vl_points
        return [(int(round(x1*W)), int(round(y1*H))),
                (int(round(x2*W)), int(round(y2*H)))]
    (x1, y1), (x2, y2) = vl_points
    return [(int(round(x1)), int(round(y1))), (int(round(x2)), int(round(y2)))]

# ---------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--override", nargs="*", default=[], help="key=value overrides")
    args = ap.parse_args()

    print("[INFO] Script started")
    cfg = load_config(args.config)
    for kv in args.override:
        k, v = parse_override(kv); set_in_dict(cfg, k, v)

    # --- config ---
    model_path = cfg["detection_model"]                          # LP model (plates)
    device = cfg.get("device", "cpu")
    source = cfg.get("video_source", 0)
    conf = float(cfg.get("confidence", 0.25))
    iou_thr = float(cfg.get("iou", 0.45))                         # YOLO IoU threshold
    max_det = int(cfg.get("max_det", 200))
    blur_cfg = cfg.get("blur", {"type": "mosaic", "pixel_size": 20})

    save_video = bool(cfg.get("save_video", True))
    save_path = cfg.get("save_path", "outputs/annotated.mp4")
    save_fps = int(cfg.get("save_fps", 25))
    output_speed = float(cfg.get("output_speed", 1.0))            # slow-motion factor
    save_fps = max(1, int(save_fps * output_speed))

    log_json = bool(cfg.get("log_json", True))
    log_path = cfg.get("log_path", "outputs/detections.json")
    show_window = bool(cfg.get("show_window", False))

    # overlays
    overlay_cfg = cfg.get("overlay", {"show_time": True, "font_scale": 0.9})
    overlay_show_time = bool(overlay_cfg.get("show_time", True))
    overlay_font_scale = float(overlay_cfg.get("font_scale", 0.9))

    # OCR config
    ocr_cfg = cfg.get("ocr", {})
    ocr_enabled = bool(ocr_cfg.get("enabled", True))
    ocr_upper = bool(ocr_cfg.get("uppercase", True))
    ocr_alnum = bool(ocr_cfg.get("alnum_only", True))
    ocr_minlen = int(ocr_cfg.get("min_text_len", 4))
    quality_threshold = float(ocr_cfg.get("quality_threshold", 120.0))

    ocr = OCR(
        engine=ocr_cfg.get("engine", "rapidocr"),
        languages=ocr_cfg.get("languages", ["en"]),
        uppercase=ocr_upper, alnum_only=ocr_alnum, min_text_len=ocr_minlen
    ) if ocr_enabled else None

    # captures / export
    caps = cfg.get("captures", {})
    plates_dir = caps.get("plates_dir", "outputs/captures/plates")
    plates_enh_dir = os.path.join(plates_dir, "enhanced")
    vehicles_dir = caps.get("vehicles_dir", "outputs/captures/vehicles")
    ann_dir = caps.get("annotated_dir", "outputs/captures/annotated")
    for d in (plates_dir, plates_enh_dir, vehicles_dir, ann_dir, os.path.dirname(save_path), os.path.dirname(log_path)):
        os.makedirs(d, exist_ok=True)

    exp = cfg.get("export", {})
    csv_path = exp.get("csv_path", "outputs/plate_events.csv")
    json_events_path = exp.get("json_path", "outputs/plate_events.json")

    # vehicle crop/overlay
    pad_ratio = float(cfg.get("vehicle_crop", {"pad_ratio": 2.5}).get("pad_ratio", 2.5))

    # tracking / debounce
    tr_cfg = cfg.get("tracker", {"enabled": True, "debounce_seconds": 3.0, "max_age": 30, "min_iou": 0.2})
    tracker = TrackManager(
        enabled=bool(tr_cfg.get("enabled", True)),
        debounce_seconds=float(tr_cfg.get("debounce_seconds", 3.0)),
        max_age=int(tr_cfg.get("max_age", 30)),
        min_iou=float(tr_cfg.get("min_iou", 0.2))
    )

    # --- virtual line (counting) ---
    vl_cfg = cfg.get("virtual_line", {})
    vl_enabled = bool(vl_cfg.get("enabled", False))
    vl_points_cfg = vl_cfg.get("points", [[100, 600], [1200, 600]])
    vl_normalized = bool(vl_cfg.get("normalized", False))
    vl_y_only = vl_cfg.get("y", None)  # optional horizontal line shortcut
    vl_color = (0, 0, 255)             # red (as you requested)
    vl_thickness = int(vl_cfg.get("thickness", 4))
    inbound_label = vl_cfg.get("in_label", "Inbound")
    outbound_label = vl_cfg.get("out_label", "Outbound")
    vl_counts = {"total": 0, "inbound": 0, "outbound": 0}
    crossed_tracks = set()
    active_texts = []  # [(plate, x, y, t0)]

    def crossing_direction(prev_y, curr_y, line_y):
        if prev_y > line_y >= curr_y:
            return "inbound"
        if prev_y < line_y <= curr_y:
            return "outbound"
        return ""

    # --- secondary vehicle detector (for ALL vehicles) ---
    sec_cfg = cfg.get("secondary_detector", {})
    sec_model_path = sec_cfg.get("model", "yolov8n.pt")  # default small COCO model if present
    sec_conf = float(sec_cfg.get("confidence", 0.25))
    sec_iou_thr = float(sec_cfg.get("iou", 0.5))
    VEH_CLASSES = set(sec_cfg.get("vehicle_class_ids", [2, 3, 5, 7]))  # car, motorcycle, bus, truck

    print(f"[INFO] Loading YOLO LP model: {model_path}")
    lp_model = YOLO(model_path)
    veh_model = YOLO(sec_model_path) if sec_model_path else None
    if veh_model:
        print(f"[INFO] Loading secondary vehicle model: {sec_model_path}")

    # source
    if isinstance(source, str) and source.isdigit():
        source = int(source)

    print(f"[INFO] Opening source: {source}")
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open source: {source}")

    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or save_fps
    tracker.set_fps(src_fps)

    # scaled output
    out_scale = float(cfg.get("output_scale", 0.5))
    OW, OH = int(W * out_scale), int(H * out_scale)
    sink = VideoSink(save_path, (OW, OH), save_fps) if save_video else None
    if sink:
        print(f"[INFO] VideoSink → {save_path} at {save_fps} FPS (speed factor {output_speed})")

    # annotators
    plate_annot = sv.BoxAnnotator(color=sv.Color.from_hex("#00FF00"))
    veh_annot   = sv.BoxAnnotator(color=sv.Color.from_hex("#00C8FF"))

    detections_log, events = [], []
    crossing_events = []
    frame_idx, processed = 0, 0
    t0_wall = time.time()
    recent_plates = collections.deque(maxlen=int(cfg.get("recent_list_size", 6)))

    # NEW: remember last readable plate per track id (persists across frames)
    last_plate_by_track = {}

    # --------- NEW: robust line-crossing helpers/state (added) ----------
    def signed_side(p1, p2, cx, cy):
        (x1, y1), (x2, y2) = p1, p2
        return (x2 - x1) * (cy - y1) - (y2 - y1) * (cx - x1)

    vl_eps = float(vl_cfg.get("deadband", 6.0))            # jitter tolerance in pixels
    vl_force_direction = vl_cfg.get("force_direction", "") # "", "inbound", or "outbound"
    veh_prev_side = {}   # veh_id -> last signed side
    lp_prev_side  = {}   # track_id -> last signed side
    # -------------------------------------------------------------------

    print("[INFO] Processing... Press ESC to stop if show_window=True")

    def timecode_from_frame(i):
        sec = 0.0 if src_fps == 0 else i / src_fps
        hh, mm, ss = int(sec // 3600), int((sec % 3600) // 60), sec % 60
        return f"{hh:02d}:{mm:02d}:{ss:06.3f}", sec

    while True:
        ok, frame = cap.read()
        if not ok: break
        processed += 1

        # 1) Global plate pass
        lp_res = lp_model.predict(frame, conf=conf, iou=iou_thr, max_det=max_det, device=device, verbose=False)[0]
        b = lp_res.boxes
        lp_boxes = b.xyxy.cpu().numpy().astype(int) if (b is not None and b.xyxy is not None) else []
        lp_confs = b.conf.cpu().numpy().tolist() if (b is not None and b.conf is not None) else []

        # 2) Vehicle detections – ensure we label every vehicle
        veh_boxes = []
        if veh_model is not None:
            vr = veh_model.predict(frame, conf=sec_conf, iou=sec_iou_thr, max_det=max_det, device=device, verbose=False)[0]
            if vr and vr.boxes is not None:
                vb = vr.boxes
                v_xyxy = vb.xyxy.cpu().numpy().astype(int)
                v_cls = vb.cls.cpu().numpy().astype(int)
                for i, xyxy in enumerate(v_xyxy):
                    if i < len(v_cls) and v_cls[i] in VEH_CLASSES:
                        veh_boxes.append(xyxy.tolist())

        # 3) Build plate ROIs with OCR/clarity (from global pass)
        rois, labels, texts, clarities, sharp_vals = [], [], [], [], []
        for i, (x1, y1, x2, y2) in enumerate(lp_boxes):
            x1 = clip(x1, 0, W - 1); x2 = clip(x2, 0, W - 1)
            y1 = clip(y1, 0, H - 1); y2 = clip(y2, 0, H - 1)
            if x2 <= x1 or y2 <= y1: continue

            txt, clarity, sharpness = "", "CLEAR", 0.0
            crop = frame[y1:y2, x1:x2]
            if ocr and crop.size > 0:
                sharpness = measure_sharpness(crop)
                if sharpness < quality_threshold:
                    clarity = "BLURRY"
                    txt = clean_text(ocr.infer(enhance_plate(crop)), ocr_upper, ocr_alnum)
                    if not txt: clarity = "NOT CLEAR"
                else:
                    txt = clean_text(ocr.infer(crop), ocr_upper, ocr_alnum)
                    if not txt: clarity = "NOT CLEAR"

            rois.append([x1, y1, x2, y2])
            label = (txt if txt else "LP") + (f" {lp_confs[i]:.2f}" if i < len(lp_confs) else "")
            labels.append(label); texts.append(txt)
            clarities.append(clarity); sharp_vals.append(float(sharpness))

        # 4) Vehicle → plate search (inside each vehicle; ensures we don’t miss)
        veh_infos = []  # dicts: {xyxy, plate_xyxy or None, plate_text or "UNREADABLE"}
        search_conf = max(0.10, conf*0.5)
        for vx1, vy1, vx2, vy2 in veh_boxes:
            # pad a bit to catch plate near bumper edges
            px1 = max(0, int(vx1 - 0.02*(vx2-vx1))); py1 = max(0, int(vy1 - 0.02*(vy2-vy1)))
            px2 = min(W-1, int(vx2 + 0.02*(vx2-vx1))); py2 = min(H-1, int(vy2 + 0.02*(vy2-vy1)))
            veh_crop = frame[py1:py2, px1:px2]
            best_plate = None
            plate_text = ""
            if veh_crop.size > 0:
                sub = lp_model.predict(
                    veh_crop, conf=search_conf, iou=iou_thr, max_det=5, device=device, verbose=False
                )[0]
                if sub and sub.boxes is not None and len(sub.boxes) > 0:
                    sb = sub.boxes
                    s_xyxy = sb.xyxy.cpu().numpy().astype(int)
                    s_conf = sb.conf.cpu().numpy().tolist()
                    j = int(np.argmax(s_conf))
                    bx1, by1, bx2, by2 = s_xyxy[j]
                    # map back to full-frame
                    gx1, gy1, gx2, gy2 = bx1+px1, by1+py1, bx2+px1, by2+py1
                    best_plate = [int(gx1), int(gy1), int(gx2), int(gy2)]

                    # OCR it
                    pcrop = frame[gy1:gy2, gx1:gx2]
                    if ocr and pcrop.size > 0:
                        sharpness = measure_sharpness(pcrop)
                        if sharpness < quality_threshold:
                            txt = clean_text(ocr.infer(enhance_plate(pcrop)), ocr_upper, ocr_alnum)
                        else:
                            txt = clean_text(ocr.infer(pcrop), ocr_upper, ocr_alnum)
                        plate_text = txt or "UNREADABLE"
                    else:
                        plate_text = "UNREADABLE"
                else:
                    plate_text = "UNREADABLE"
            else:
                plate_text = "UNREADABLE"

            # merge this plate into global rois (avoid duplicates via IoU > 0.5)
            if best_plate is not None:
                dup = any(bbox_iou(best_plate, r) > 0.5 for r in rois)
                if not dup:
                    rois.append(best_plate)
                    texts.append(plate_text if plate_text != "UNREADABLE" else "")
                    clarities.append("CLEAR" if plate_text != "UNREADABLE" else "NOT CLEAR")
                    sharp_vals.append(0.0)
                    labels.append(plate_text if plate_text != "UNREADABLE" else "LP")

            veh_infos.append({"xyxy":[vx1,vy1,vx2,vy2], "plate":best_plate, "text":plate_text})

        # 5) Update LP tracks (debounced events per track)
        tracks = tracker.update(rois, texts, frame_idx)

        # ---- Plate persistence per track (remember last readable) ----
        for tr in tracks:
            tid = tr["id"]
            ptxt = (tr.get("plate_text") or "").strip()
            if ptxt:
                last_plate_by_track[tid] = ptxt
            else:
                if tid in last_plate_by_track:
                    tr["plate_text"] = last_plate_by_track[tid]

        # 6) For counting, use vehicle boxes (covers all vehicles)
        count_boxes = veh_boxes[:] if veh_boxes else rois[:]

        # blur plates
        out = apply_blur_to_regions(frame.copy(), rois, blur_cfg)

        # draw plate boxes (green)
        if rois:
            det = sv.Detections(xyxy=np.asarray(rois, dtype=np.int32).reshape(-1,4),
                                class_id=np.zeros(len(rois), int))
            out = plate_annot.annotate(out, det)

        # draw vehicle boxes + label with plate text / UNREADABLE (cyan)
        if veh_infos:
            vdet = sv.Detections(xyxy=np.asarray([v["xyxy"] for v in veh_infos], dtype=np.int32),
                                    class_id=np.zeros(len(veh_infos), int))
            out = veh_annot.annotate(out, vdet)
            font = cv2.FONT_HERSHEY_SIMPLEX
            for v in veh_infos:
                vx1, vy1, vx2, vy2 = map(int, v["xyxy"])
                ptxt = v["text"] or "UNREADABLE"
                banner = f"PLATE: {ptxt}"
                (tw, th), _ = cv2.getTextSize(banner, font, 0.7, 2)
                y_top = max(0, vy1 - th - 8)
                cv2.rectangle(out, (vx1, y_top), (vx1+tw+8, y_top+th+6), (255, 255, 255), -1)
                cv2.putText(out, banner, (vx1+4, y_top+th+1), font, 0.7, (0,0,0), 2, cv2.LINE_AA)

        # --- virtual line crossing (updated: robust signed-side + stable IDs + persistence) ---
        if vl_enabled:
            p1, p2 = resolve_line_points(vl_points_cfg, W, H, normalized=vl_normalized, y_only=vl_y_only)

            # 1) Plate-track crossings (use track IDs)
            for tr in tracks:
                tx1, ty1, tx2, ty2 = map(int, tr["xyxy"])
                cx, cy = int((tx1 + tx2) / 2), int((ty1 + ty2) / 2)
                curr = signed_side(p1, p2, cx, cy)

                tid = tr["id"]
                prev = lp_prev_side.get(tid, curr)
                lp_prev_side[tid] = curr

                # ignore jitter near the line
                if abs(prev) < vl_eps or abs(curr) < vl_eps:
                    continue

                # detect single crossing on sign change
                if (prev > 0 and curr < 0) or (prev < 0 and curr > 0):
                    dirn = "inbound" if prev > 0 and curr < 0 else "outbound"
                    if vl_force_direction in ("inbound", "outbound"):
                        dirn = vl_force_direction

                    if ("lp", tid) not in crossed_tracks:
                        crossed_tracks.add(("lp", tid))
                        vl_counts["total"] += 1
                        vl_counts[dirn] += 1

                        plate_num = (tr.get("plate_text") or "").strip() or "UNREADABLE"
                        active_texts.append((plate_num, cx, int((p1[1] + p2[1]) / 2) - 18, time.time()))
                        tcode, _ = timecode_from_frame(frame_idx)
                        crossing_events.append({
                            "track_id": tid,
                            "plate_text": ("" if plate_num == "UNREADABLE" else plate_num),
                            "unreadable": (plate_num == "UNREADABLE"),
                            "direction": dirn,
                            "frame_index": frame_idx,
                            "timecode_video": tcode,
                            "time_wall": datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                        })

            # 2) Vehicle-box crossings (fallback for vehicles without plate tracks)
            # Create stable veh IDs via nearest-neighbor across frames
            veh_centers = []
            for bb in count_boxes:
                vx1, vy1, vx2, vy2 = bb
                veh_centers.append(((vx1 + vx2) / 2.0, (vy1 + vy2) / 2.0))

            if not hasattr(tracker, "veh_last_centers"):
                tracker.veh_last_centers = []
                tracker.veh_ids = []
                tracker.next_veh_id = 1

            used_last = set()
            curr_ids = []
            for cx, cy in veh_centers:
                best_k, best_d2 = -1, 1e12
                for k, (lx, ly) in enumerate(tracker.veh_last_centers):
                    if k in used_last:
                        continue
                    d2 = (cx - lx) ** 2 + (cy - ly) ** 2
                    if d2 < best_d2:
                        best_d2 = d2
                        best_k = k
                if best_k >= 0 and best_d2 < 200**2:
                    curr_ids.append(tracker.veh_ids[best_k])
                    used_last.add(best_k)
                else:
                    curr_ids.append(tracker.next_veh_id)
                    tracker.next_veh_id += 1

            tracker.veh_last_centers = veh_centers
            tracker.veh_ids = curr_ids

            # short persistence buffer to tolerate brief detector dropouts
            if not hasattr(tracker, "veh_seen_last"):
                tracker.veh_seen_last = {}

            new_seen = {}
            for idx, bb in enumerate(count_boxes):
                if idx >= len(tracker.veh_ids):
                    continue
                veh_id = tracker.veh_ids[idx]
                vx1, vy1, vx2, vy2 = bb
                cx, cy = int((vx1 + vx2) / 2), int((vy1 + vy2) / 2)
                curr = signed_side(p1, p2, cx, cy)
                prev = veh_prev_side.get(veh_id, curr)
                veh_prev_side[veh_id] = curr
                new_seen[veh_id] = frame_idx

                if abs(prev) < vl_eps or abs(curr) < vl_eps:
                    continue

                if (prev > 0 and curr < 0) or (prev < 0 and curr > 0):
                    dirn = "inbound" if prev > 0 and curr < 0 else "outbound"
                    if vl_force_direction in ("inbound", "outbound"):
                        dirn = vl_force_direction

                    if ("veh", veh_id) not in crossed_tracks:
                        crossed_tracks.add(("veh", veh_id))
                        vl_counts["total"] += 1
                        vl_counts[dirn] += 1

                        ptxt = "UNREADABLE"
                        if idx < len(veh_infos):
                            ptxt = veh_infos[idx]["text"] or "UNREADABLE"

                        active_texts.append((ptxt, cx, int((p1[1] + p2[1]) / 2) - 18, time.time()))
                        tcode, _ = timecode_from_frame(frame_idx)
                        crossing_events.append({
                            "track_id": f"veh_{veh_id}",
                            "plate_text": ("" if ptxt == "UNREADABLE" else ptxt),
                            "unreadable": (ptxt == "UNREADABLE"),
                            "direction": dirn,
                            "frame_index": frame_idx,
                            "timecode_video": tcode,
                            "time_wall": datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                        })

            # purge vehicles not seen for > N frames (prevents stale state)
            N = 5
            for vid, lastf in list(tracker.veh_seen_last.items()):
                if frame_idx - lastf > N:
                    veh_prev_side.pop(vid, None)
            tracker.veh_seen_last = new_seen

        # helper: map track to closest roi index
        def best_match_idx(track_xyxy):
            if not rois: return -1, 0.0
            best_k, best_v = -1, 0.0
            for k,r in enumerate(rois):
                v = bbox_iou(track_xyxy, r)
                if v > best_v: best_v, best_k = v, k
            return best_k, best_v

        # draw plate text & track id banners (with clarity)
        font = cv2.FONT_HERSHEY_SIMPLEX
        for tr in tracks:
            x1, y1, x2, y2 = map(int, tr["xyxy"])
            k, _ = best_match_idx(tr["xyxy"])
            plate = tr.get("plate_text", "")
            clarity = clarities[k] if (0 <= k < len(clarities)) else ""
            label = (plate if plate else "UNREADABLE")
            color = (0,255,0) if clarity == "CLEAR" else ((0,165,255) if clarity=="BLURRY" else (0,0,255))
            label_full = f"ID {tr['id']} | {label}" + (f" ({clarity})" if clarity else "")
            (tw, th), _ = cv2.getTextSize(label_full, font, overlay_font_scale, 2)
            y_top = max(0, y1 - th - 10)
            rect_w = max(1, tw + 10); rect_h = max(1, th + 6)
            cv2.rectangle(out, (int(x1), int(y_top)), (int(x1 + rect_w), int(y_top + rect_h)), color, -1)
            cv2.putText(out, label_full, (int(x1 + 4), int(y_top + th + 1)),
                        font, overlay_font_scale, (0, 0, 0), 2, cv2.LINE_AA)

        # time overlay
        if overlay_show_time:
            timecode, _ = timecode_from_frame(frame_idx)
            cv2.putText(out, timecode, (12, 28), font, 0.8, (0, 255, 255), 2, cv2.LINE_AA)

        # --- draw virtual line & counters & floating plate popups ---
        if vl_enabled:
            # p1,p2 already computed in the crossing block
            cv2.line(out, p1, p2, vl_color, vl_thickness)
            header = f"Total: {vl_counts['total']} | {inbound_label}: {vl_counts['inbound']} | {outbound_label}: {vl_counts['outbound']}"
            # BIGGER/clearer counter text (scale 1.3, thickness 3)
            cv2.putText(out, header,
                        (int(p1[0] + 10), max(0, int(p1[1] - 20))),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255,255,255), 3, cv2.LINE_AA)

            now_t = time.time()
            active_texts = [(p,x,y,t0) for (p,x,y,t0) in active_texts if now_t - t0 < 2.0]
            for (plate, x, y, t0) in active_texts:
                fade = max(0, 1 - (now_t - t0)/2.0)
                color = (0, int(255*fade), int(255*fade))
                cv2.putText(out, plate, (int(x-60), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3, cv2.LINE_AA)

        # --- top-right recent plates panel (only readable) ---
        for tr in tracks:
            plate_num = (tr.get("plate_text") or "").strip()
            if plate_num and tr.get("fired", False) and (not recent_plates or recent_plates[-1] != plate_num):
                recent_plates.append(plate_num)

        panel_w, row_h = 320, 30
        x0 = (out.shape[1]) - panel_w - 10
        y0 = 10
        cv2.rectangle(out, (int(x0), int(y0)), (int(x0 + panel_w), int(y0 + 12 + row_h*(len(recent_plates)+1))),
                        (20,20,20), -1)
        cv2.putText(out, "Recent Plates", (int(x0+10), int(y0+25)), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (255,255,255), 2, cv2.LINE_AA)
        for i, ptxt in enumerate(recent_plates):
            cv2.putText(out, f"{i+1}. {ptxt}", (int(x0+10), int(y0+25 + (i+1)*row_h)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180,255,180), 2, cv2.LINE_AA)

        # write video (downscaled)
        if sink:
            out_small = cv2.resize(out, (OW, OH)) if (OW != out.shape[1] or OH != out.shape[0]) else out
            sink.write(out_small)

        # build events for "fired" tracks (debounced) – same as before
        timecode, sec = timecode_from_frame(frame_idx)
        wall_clock = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        for tr in tracks:
            if not tr.get("fired"): continue
            x1, y1, x2, y2 = map(int, tr["xyxy"])

            # associate to nearest roi
            k = -1; best = 0.0
            for i, r in enumerate(rois):
                v = bbox_iou(tr["xyxy"], r)
                if v > best: best, k = v, i

            plate_text = texts[k] if (0 <= k < len(texts)) else tr.get("plate_text", "")
            if (not plate_text) and last_plate_by_track.get(tr["id"]):
                plate_text = last_plate_by_track[tr["id"]]
            clarity = clarities[k] if (0 <= k < len(clarities)) else ""
            sharpness = sharp_vals[k] if (0 <= k < len(sharp_vals)) else 0.0

            vx1, vy1, vx2, vy2 = expand_box(x1, y1, x2, y2, pad_ratio, W, H)
            plate_crop = frame[y1:y2, x1:x2]
            vehicle_crop = frame[vy1:vy2, vx1:vx2]
            plate_enh = enhance_plate(plate_crop) if (clarity in ("BLURRY","NOT CLEAR")) else None

            stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            base = f"{stamp}_f{frame_idx}_id{tr['id']}"
            p_path = os.path.join(plates_dir, f"{base}_plate.jpg")
            pe_path = os.path.join(plates_enh_dir, f"{base}_plate_enh.jpg")
            v_path = os.path.join(vehicles_dir, f"{base}_vehicle.jpg")
            a_path = os.path.join(ann_dir, f"{base}_annotated.jpg")

            if plate_crop.size > 0: cv2.imwrite(p_path, plate_crop)
            if plate_enh is not None and plate_enh.size > 0: cv2.imwrite(pe_path, plate_enh)
            if vehicle_crop.size > 0: cv2.imwrite(v_path, vehicle_crop)
            cv2.imwrite(a_path, out)

            plate_hash = hashlib.sha256((plate_text or "").encode("utf-8")).hexdigest()[:10] if plate_text else ""
            events.append({
                "track_id": tr["id"], "timecode_video": timecode, "time_wall": wall_clock,
                "frame_index": frame_idx, "plate_text": plate_text, "plate_hash": plate_hash,
                "clarity_status": clarity, "sharpness_value": round(float(sharpness), 2),
                "confidence": None, "plate_bbox_xyxy": [x1, y1, x2, y2],
                "vehicle_bbox_xyxy": [vx1, vy1, vx2, vy2],
                "plate_image": p_path.replace("\\", "/"),
                "plate_image_enhanced": (pe_path.replace("\\", "/") if os.path.exists(pe_path) else ""),
                "vehicle_image": v_path.replace("\\", "/"),
                "annotated_image": a_path.replace("\\", "/")
            })

        if log_json and rois:
            detections_log.append({"frame": frame_idx, "timecode": timecode, "n_boxes": len(rois)})

        frame_idx += 1
        if show_window:
            cv2.imshow("LP Blur", out)
            if cv2.waitKey(1) & 0xFF == 27: break

    # cleanup
    cap.release()
    if show_window: cv2.destroyAllWindows()
    if sink: sink.release()

    # exports (plate events)
    if events:
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        header = [
            "track_id","timecode_video","time_wall","frame_index",
            "plate_text","plate_hash","clarity_status","sharpness_value",
            "confidence","plate_bbox_xyxy","vehicle_bbox_xyxy",
            "plate_image","plate_image_enhanced","vehicle_image","annotated_image"
        ]
        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=header)
            if f.tell()==0: w.writeheader()
            w.writerows(events)
        with open(json_events_path,"w",encoding="utf-8") as f:
            json.dump(events,f,indent=2)

    # exports (crossing events)
    cross_csv = "outputs/crossings.csv"
    cross_json = "outputs/crossings.json"
    if crossing_events:
        os.makedirs("outputs", exist_ok=True)
        with open(cross_csv, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["track_id","plate_text","unreadable","direction","frame_index","timecode_video","time_wall"])
            if f.tell()==0: w.writeheader()
            w.writerows(crossing_events)
        with open(cross_json, "w", encoding="utf-8") as f:
            json.dump(crossing_events, f, indent=2)

    # simple analytics summary
    analytics_out = cfg.get("analytics", {}).get("report_json", "outputs/analytics_summary.json")
    summary = {"video_frames_processed": processed, "duration_s": round(time.time() - t0_wall, 3)}
    if events:
        summary.update({
            "total_events": len(events),
            "unique_plates": len({e["plate_text"] for e in events if e["plate_text"]})
        })
    if crossing_events:
        by_dir = {"inbound": 0, "outbound": 0}
        for e in crossing_events:
            by_dir[e["direction"]] = by_dir.get(e["direction"], 0) + 1
        summary.update({
            "total_crossings": len(crossing_events),
            "crossings_by_direction": by_dir
        })
    os.makedirs(os.path.dirname(analytics_out), exist_ok=True)
    with open(analytics_out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"[INFO] Frames processed: {processed}")
    print(f"[INFO] Plate events saved: {len(events)}")
    print(f"[INFO] Crossing events saved: {len(crossing_events)}")
    print(f"[INFO] Video saved to: {sink.path if sink else '(no video)'}")
    print("[INFO] Done. Check outputs/ for results.")

if __name__ == "__main__":
    main()
