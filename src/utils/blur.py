# Blurring utilities
import cv2

def apply_blur_to_regions(image, rois, blur_cfg):
    t = (blur_cfg or {}).get("type", "mosaic").lower()
    if t == "gaussian":
        k = int((blur_cfg or {}).get("kernel", 25))
        if k % 2 == 0:
            k += 1
        sigma = float((blur_cfg or {}).get("sigma", 0))
        for x1, y1, x2, y2 in rois:
            y1, y2 = max(0, y1), min(image.shape[0], y2)
            x1, x2 = max(0, x1), min(image.shape[1], x2)
            if x2 > x1 and y2 > y1:
                image[y1:y2, x1:x2] = cv2.GaussianBlur(image[y1:y2, x1:x2], (k, k), sigma)
        return image
    else:
        # mosaic / pixelate
        pix = int((blur_cfg or {}).get("pixel_size", 20))
        for x1, y1, x2, y2 in rois:
            y1, y2 = max(0, y1), min(image.shape[0], y2)
            x1, x2 = max(0, x1), min(image.shape[1], x2)
            if x2 > x1 and y2 > y1:
                roi = image[y1:y2, x1:x2]
                h, w = roi.shape[:2]
                small = cv2.resize(roi, (max(1, w // pix), max(1, h // pix)), interpolation=cv2.INTER_LINEAR)
                image[y1:y2, x1:x2] = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
        return image
