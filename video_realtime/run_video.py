"""Chạy model phát hiện + tracking + đếm xe trên video.

Ví dụ:
    .venv/Scripts/python.exe video_realtime/run_video.py --source video_realtime/a.mp4
    .venv/Scripts/python.exe video_realtime/run_video.py --source 0 --show      # webcam
"""
import argparse
import time
from collections import defaultdict
from pathlib import Path

import cv2
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WEIGHTS = ROOT / "runs/vehicle_v1_yolo26s/weights/best.pt"
COLORS = {0: (255, 160, 0), 1: (0, 200, 0), 2: (0, 140, 255), 3: (200, 0, 200)}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--source", required=True, help="đường dẫn video hoặc chỉ số webcam (0)")
    p.add_argument("--weights", default=str(DEFAULT_WEIGHTS))
    p.add_argument("--out", default=None, help="file video kết quả (mặc định: <tên>_out.mp4)")
    p.add_argument("--conf", type=float, default=0.35)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--line", type=float, default=0.6, help="vị trí vạch đếm theo chiều cao (0-1)")
    p.add_argument("--tracker", default="bytetrack.yaml")
    p.add_argument("--show", action="store_true", help="hiện cửa sổ xem trực tiếp")
    return p.parse_args()


def main():
    a = parse_args()
    src = int(a.source) if a.source.isdigit() else a.source
    if not isinstance(src, int) and (not Path(src).exists() or Path(src).stat().st_size == 0):
        raise SystemExit(f"Video không tồn tại hoặc rỗng: {src}")

    model = YOLO(a.weights)
    names = model.names

    cap = cv2.VideoCapture(src)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    line_y = int(h * a.line)

    out_path = a.out or (str(Path(str(a.source)).with_suffix("")) + "_out.mp4" if not isinstance(src, int) else "webcam_out.mp4")
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    last_y = {}                       # track_id -> tâm y ở frame trước
    counted = set()                   # track_id đã đếm
    counts = defaultdict(int)         # class -> số xe qua vạch
    n_frames, t0 = 0, time.time()

    for r in model.track(src, stream=True, persist=True, conf=a.conf, imgsz=a.imgsz,
                         tracker=a.tracker, verbose=False):
        frame = r.orig_img.copy()
        n_frames += 1
        if r.boxes is not None and r.boxes.id is not None:
            for (x1, y1, x2, y2), cls, tid, cf in zip(r.boxes.xyxy.int().tolist(), r.boxes.cls.int().tolist(),
                                                     r.boxes.id.int().tolist(), r.boxes.conf.tolist()):
                cy = (y1 + y2) // 2
                py = last_y.get(tid)
                if py is not None and tid not in counted and (py - line_y) * (cy - line_y) <= 0 and py != cy:
                    counted.add(tid)
                    counts[cls] += 1
                last_y[tid] = cy
                c = COLORS.get(cls, (255, 255, 255))
                cv2.rectangle(frame, (x1, y1), (x2, y2), c, 2)
                cv2.putText(frame, f"{names[cls]} #{tid} {cf:.2f}", (x1, max(y1 - 6, 12)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 2)

        cv2.line(frame, (0, line_y), (w, line_y), (0, 0, 255), 2)
        live_fps = n_frames / (time.time() - t0)
        lines = [f"FPS: {live_fps:.1f}", f"Tong: {sum(counts.values())}"] + \
                [f"{names[k]}: {counts[k]}" for k in sorted(names)]
        cv2.rectangle(frame, (8, 8), (200, 18 + 24 * len(lines)), (0, 0, 0), -1)
        for i, t in enumerate(lines):
            cv2.putText(frame, t, (16, 32 + 24 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        writer.write(frame)
        if a.show:
            cv2.imshow("traffic", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    writer.release()
    cv2.destroyAllWindows()
    print(f"Xong {n_frames} frame, {n_frames / (time.time() - t0):.1f} FPS trung bình")
    print("Đếm qua vạch:", {names[k]: v for k, v in counts.items()}, "| tổng", sum(counts.values()))
    print("Video kết quả:", out_path)


if __name__ == "__main__":
    main()
