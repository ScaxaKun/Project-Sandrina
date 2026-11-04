from ultralytics import YOLO
import cv2
import numpy as np
import time
import os
from collections import deque, Counter
from PIL import Image, ImageDraw, ImageFont


# =========================================================
# 1. LOAD MODEL
# =========================================================

detector = YOLO("yolo26n.pt")
classifier = YOLO("runs/polar_3class/weights/best.pt")


# =========================================================
# 2. WEBCAM
# =========================================================

cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

if not cap.isOpened():
    print("Kamera tidak bisa dibuka.")
    raise SystemExit


# =========================================================
# 3. DASHBOARD CONFIGURATION
# =========================================================

CAM_W = 1020
CAM_H = 700

SIDEBAR_W = 480
HEADER_H = 76
FOOTER_H = 40

WINDOW_NAME = "CCTV AI - Mini Dashboard"

cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

cv2.resizeWindow(
    WINDOW_NAME,
    CAM_W + SIDEBAR_W,
    CAM_H + HEADER_H + FOOTER_H
)


# =========================================================
# 4. ACTIVITY AND ALERT CONFIGURATION
# =========================================================

SITTING_LIMIT_SECONDS = 30 * 60
STANDING_LIMIT_SECONDS = 60 * 60

TRACK_TIMEOUT_SECONDS = 3

prediction_history = {}
current_status = {}
status_start_time = {}
last_seen_time = {}

activity_history = deque(maxlen=100)

show_history = False
show_alerts = False


# =========================================================
# 5. COLORS (BGR)
# =========================================================

BG_NAVY = (39, 24, 17)
HEADER_NAVY = (59, 41, 30)
SIDEBAR_NAVY = (48, 31, 21)
CARD_NAVY = (71, 50, 36)

TEXT_WHITE = (255, 255, 255)
TEXT_MUTED = (215, 222, 232)

TEAL = (165, 205, 105)

RED = (70, 85, 230)
ORANGE = (40, 165, 240)

DIVIDER = (86, 68, 55)


# =========================================================
# 6. FONT CONFIGURATION
# =========================================================

FONT_CANDIDATES = {
    "regular": [
        "fonts/Regular.ttf",
        "C:/Windows/Fonts/segoeui.ttf",
        "C:/Windows/Fonts/arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    ],

    "semibold": [
        "fonts/SemiBold.ttf",
        "C:/Windows/Fonts/seguisb.ttf",
        "C:/Windows/Fonts/arialbd.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
    ],

    "bold": [
        "fonts/Bold.ttf",
        "C:/Windows/Fonts/segoeuib.ttf",
        "C:/Windows/Fonts/arialbd.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
    ]
}

_font_cache = {}


def get_font(size, weight):
    key = (size, weight)

    if key in _font_cache:
        return _font_cache[key]

    font = None

    for path in FONT_CANDIDATES[weight]:
        if os.path.exists(path):
            try:
                font = ImageFont.truetype(path, size)
                break
            except OSError:
                continue

    if font is None:
        print(
            "Peringatan: font TrueType tidak ditemukan, "
            "menggunakan font bawaan."
        )
        font = ImageFont.load_default()

    _font_cache[key] = font

    return font


# =========================================================
# 7. PILLOW TEXT RENDERING
# =========================================================

text_queue = []


def draw_text(
    text,
    position,
    size=16,
    color=TEXT_WHITE,
    weight="regular",
    anchor="ls",
    stroke=0,
    stroke_color=(0, 0, 0)
):
    text_queue.append(
        (
            str(text),
            position,
            size,
            color[::-1],
            weight,
            anchor,
            stroke,
            stroke_color[::-1]
        )
    )


def render_text(image_bgr):
    if not text_queue:
        return image_bgr

    pil_image = Image.fromarray(
        cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    )

    pen = ImageDraw.Draw(pil_image)

    for (
        text,
        position,
        size,
        color,
        weight,
        anchor,
        stroke,
        stroke_color
    ) in text_queue:

        pen.text(
            position,
            text,
            font=get_font(size, weight),
            fill=color,
            anchor=anchor,
            stroke_width=stroke,
            stroke_fill=stroke_color
        )

    return cv2.cvtColor(
        np.array(pil_image),
        cv2.COLOR_RGB2BGR
    )


# =========================================================
# 8. DURATION FORMAT
# =========================================================

def format_duration(seconds):
    seconds = max(0, int(seconds))

    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60

    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"

    return f"{minutes:02d}:{secs:02d}"


# =========================================================
# 9. ALERT INFORMATION
# =========================================================

def get_alert_info(status, duration):
    normalized_status = status.lower()

    if (
        normalized_status == "sitting"
        and duration >= SITTING_LIMIT_SECONDS
    ):
        return (
            "Terlalu lama duduk",
            "Berdiri sejenak dan lakukan peregangan",
            "Ringan untuk membantu mengurangi",
            "kekakuan otot akibat duduk terlalu lama.",
            "sitting"
        )

    elif (
        normalized_status == "standing"
        and duration >= STANDING_LIMIT_SECONDS
    ):
        return (
            "Terlalu lama berdiri",
            "Duduk sejenak dan ubah posisi tubuh",
            "Untuk membantu mengurangi rasa lelah",
            "pada kaki.",
            "standing"
        )

    return None


# =========================================================
# 10. DRAW DASHBOARD HEADER
# =========================================================

def draw_header(dashboard, title, subtitle):
    canvas_w = CAM_W + SIDEBAR_W

    cv2.rectangle(
        dashboard,
        (0, 0),
        (canvas_w, HEADER_H),
        HEADER_NAVY,
        -1
    )

    draw_text(
        title,
        (22, 36),
        30,
        TEXT_WHITE,
        "bold"
    )

    draw_text(
        subtitle,
        (22, 62),
        15,
        TEXT_MUTED,
        "semibold"
    )


# =========================================================
# 11. DRAW MAIN MONITORING PAGE
# =========================================================

def draw_monitoring_page(
    dashboard,
    live_people,
    total_people,
    active_alerts
):
    sx = CAM_W

    cv2.rectangle(
        dashboard,
        (sx, HEADER_H),
        (CAM_W + SIDEBAR_W, HEADER_H + CAM_H),
        SIDEBAR_NAVY,
        -1
    )

    cv2.line(
        dashboard,
        (sx, HEADER_H),
        (sx, HEADER_H + CAM_H),
        DIVIDER,
        1
    )

    # Sidebar title
    draw_text(
        "REAL-TIME MONITORING",
        (sx + 24, HEADER_H + 35),
        22,
        TEXT_WHITE,
        "bold"
    )

    draw_text(
        "Ringkasan pemantauan aktivitas",
        (sx + 24, HEADER_H + 61),
        15,
        TEXT_MUTED
    )

    # Statistics cards
    card_y = HEADER_H + 85
    card_h = 90
    card_w = 205

    cv2.rectangle(
        dashboard,
        (sx + 20, card_y),
        (sx + 20 + card_w, card_y + card_h),
        CARD_NAVY,
        -1
    )

    cv2.rectangle(
        dashboard,
        (sx + 245, card_y),
        (sx + 245 + card_w, card_y + card_h),
        CARD_NAVY,
        -1
    )

    draw_text(
        "INDIVIDU TERDETEKSI",
        (sx + 32, card_y + 28),
        13,
        TEXT_MUTED,
        "semibold"
    )

    draw_text(
        str(total_people),
        (sx + 32, card_y + 68),
        30,
        TEXT_WHITE,
        "bold"
    )

    draw_text(
        "PERINGATAN AKTIF",
        (sx + 257, card_y + 28),
        13,
        TEXT_MUTED,
        "semibold"
    )

    draw_text(
        str(active_alerts),
        (sx + 257, card_y + 68),
        30,
        RED if active_alerts > 0 else TEAL,
        "bold"
    )

    # Individual activity section
    section_y = card_y + card_h + 38

    draw_text(
        "STATUS INDIVIDU",
        (sx + 24, section_y),
        19,
        TEXT_WHITE,
        "bold"
    )

    cv2.line(
        dashboard,
        (sx + 24, section_y + 14),
        (sx + SIDEBAR_W - 24, section_y + 14),
        DIVIDER,
        1
    )

    if not live_people:
        draw_text(
            "Belum ada individu terdeteksi.",
            (sx + 24, section_y + 55),
            16,
            TEXT_MUTED
        )

    else:
        row_y = section_y + 48

        for person_id, info in list(live_people.items())[:4]:
            status = info["status"]
            duration = format_duration(info["duration"])

            status_color = (
                RED if info["alert"] else TEAL
            )

            draw_text(
                f"ID {person_id}",
                (sx + 24, row_y),
                18,
                TEXT_WHITE,
                "bold"
            )

            draw_text(
                status.upper(),
                (sx + SIDEBAR_W - 24, row_y),
                16,
                status_color,
                "bold",
                anchor="rs"
            )

            draw_text(
                f"Durasi: {duration}",
                (sx + 24, row_y + 28),
                15,
                TEXT_MUTED
            )

            if info["alert"]:
                draw_text(
                    "PERINGATAN AKTIF",
                    (sx + SIDEBAR_W - 24, row_y + 28),
                    13,
                    RED,
                    "bold",
                    anchor="rs"
                )

            cv2.line(
                dashboard,
                (sx + 24, row_y + 42),
                (sx + SIDEBAR_W - 24, row_y + 42),
                DIVIDER,
                1
            )

            row_y += 76

    # Alert summary
    alert_section_y = HEADER_H + CAM_H - 155

    cv2.line(
        dashboard,
        (sx + 24, alert_section_y),
        (sx + SIDEBAR_W - 24, alert_section_y),
        DIVIDER,
        1
    )

    draw_text(
        "INFORMASI PERINGATAN",
        (sx + 24, alert_section_y + 30),
        17,
        TEXT_WHITE,
        "bold"
    )

    if active_alerts > 0:
        draw_text(
            f"Terdapat {active_alerts} peringatan aktif.",
            (sx + 24, alert_section_y + 58),
            15,
            RED,
            "semibold"
        )

        draw_text(
            "Tekan P untuk melihat detail peringatan.",
            (sx + 24, alert_section_y + 85),
            14,
            TEXT_MUTED
        )

    else:
        draw_text(
            "Tidak ada peringatan aktif.",
            (sx + 24, alert_section_y + 58),
            15,
            TEAL
        )


# =========================================================
# 12. DRAW ACTIVITY HISTORY PAGE
# =========================================================

def draw_history_page(dashboard):
    sx = CAM_W

    cv2.rectangle(
        dashboard,
        (sx, HEADER_H),
        (CAM_W + SIDEBAR_W, HEADER_H + CAM_H),
        SIDEBAR_NAVY,
        -1
    )

    cv2.line(
        dashboard,
        (sx, HEADER_H),
        (sx, HEADER_H + CAM_H),
        DIVIDER,
        1
    )

    draw_text(
        "RIWAYAT AKTIVITAS",
        (sx + 24, HEADER_H + 38),
        22,
        TEXT_WHITE,
        "bold"
    )

    draw_text(
        "Catatan perubahan status individu",
        (sx + 24, HEADER_H + 64),
        15,
        TEXT_MUTED
    )

    cv2.line(
        dashboard,
        (sx + 24, HEADER_H + 85),
        (sx + SIDEBAR_W - 24, HEADER_H + 85),
        DIVIDER,
        1
    )

    if not activity_history:
        draw_text(
            "Belum ada perubahan aktivitas.",
            (sx + 24, HEADER_H + 125),
            16,
            TEXT_MUTED
        )

    else:
        row_y = HEADER_H + 120

        for event in list(activity_history)[:9]:
            draw_text(
                event["time"],
                (sx + 24, row_y),
                15,
                TEXT_MUTED,
                "semibold"
            )

            draw_text(
                f"ID {event['person_id']}",
                (sx + SIDEBAR_W - 24, row_y),
                15,
                TEXT_WHITE,
                "bold",
                anchor="rs"
            )

            draw_text(
                f"{event['old_status']} -> {event['new_status']}",
                (sx + 24, row_y + 27),
                15,
                TEAL
            )

            cv2.line(
                dashboard,
                (sx + 24, row_y + 42),
                (sx + SIDEBAR_W - 24, row_y + 42),
                DIVIDER,
                1
            )

            row_y += 58


# =========================================================
# 13. DRAW ALERT PAGE
# =========================================================

def draw_alert_page(dashboard, live_people):
    sx = CAM_W

    cv2.rectangle(
        dashboard,
        (sx, HEADER_H),
        (CAM_W + SIDEBAR_W, HEADER_H + CAM_H),
        SIDEBAR_NAVY,
        -1
    )

    cv2.line(
        dashboard,
        (sx, HEADER_H),
        (sx, HEADER_H + CAM_H),
        DIVIDER,
        1
    )

    draw_text(
        "HALAMAN PERINGATAN",
        (sx + 24, HEADER_H + 38),
        22,
        TEXT_WHITE,
        "bold"
    )

    draw_text(
        "Daftar individu yang melewati batas durasi",
        (sx + 24, HEADER_H + 64),
        14,
        TEXT_MUTED
    )

    cv2.line(
        dashboard,
        (sx + 24, HEADER_H + 85),
        (sx + SIDEBAR_W - 24, HEADER_H + 85),
        DIVIDER,
        1
    )

    # Only display people with active alerts
    alert_people = {
        person_id: info
        for person_id, info in live_people.items()
        if info["alert"]
    }

    if not alert_people:
        draw_text(
            "Tidak ada peringatan aktif.",
            (sx + 24, HEADER_H + 130),
            17,
            TEAL,
            "semibold"
        )

        draw_text(
            "Individu yang terdeteksi belum melewati",
            (sx + 24, HEADER_H + 162),
            15,
            TEXT_MUTED
        )

        draw_text(
            "batas durasi aktivitas yang ditentukan.",
            (sx + 24, HEADER_H + 186),
            15,
            TEXT_MUTED
        )

        return

    row_y = HEADER_H + 125

    for person_id, info in alert_people.items():
        if row_y > HEADER_H + CAM_H - 185:
            break

        alert_type = info["alert_type"]

        cv2.rectangle(
            dashboard,
            (sx + 20, row_y - 22),
            (sx + SIDEBAR_W - 20, row_y + 165),
            CARD_NAVY,
            -1
        )

        if alert_type == "sitting":
            title = "TERLALU LAMA DUDUK"
            advice1 = "Berdiri sejenak dan lakukan peregangan."
            advice2 = "Lakukan gerakan ringan untuk membantu"
            advice3 = "mengurangi kekakuan akibat duduk lama."

        else:
            title = "TERLALU LAMA BERDIRI"
            advice1 = "Duduk sejenak dan ubah posisi tubuh."
            advice2 = "Istirahat untuk membantu mengurangi"
            advice3 = "rasa lelah pada kaki."

        draw_text(
            title,
            (sx + 35, row_y + 5),
            18,
            RED,
            "bold"
        )

        draw_text(
            f"ID Individu: {person_id}",
            (sx + 35, row_y + 35),
            15,
            TEXT_WHITE,
            "semibold"
        )

        draw_text(
            f"Aktivitas: {info['status'].upper()}",
            (sx + 35, row_y + 60),
            15,
            TEXT_WHITE
        )

        draw_text(
            f"Durasi: {format_duration(info['duration'])}",
            (sx + 35, row_y + 85),
            15,
            TEXT_WHITE
        )

        draw_text(
            advice1,
            (sx + 35, row_y + 108),
            13,
            TEXT_MUTED
        )

        draw_text(
            advice2,
            (sx + 35, row_y + 127),
            12,
            TEXT_MUTED
        )

        draw_text(
            advice3,
            (sx + 35, row_y + 146),
            12,
            TEXT_MUTED
        )

        row_y += 190


# =========================================================
# 14. MAIN PROGRAM
# =========================================================

try:
    while True:
        ret, frame = cap.read()

        if not ret:
            print("Kamera tidak bisa dibaca.")
            break

        text_queue.clear()

        frame = cv2.resize(
            frame,
            (CAM_W, CAM_H),
            interpolation=cv2.INTER_LINEAR
        )

        now = time.time()

        results = detector.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            verbose=False
        )

        result = results[0]

        current_ids = set()
        live_people = {}

        # -------------------------------------------------
        # HUMAN DETECTION AND ACTIVITY CLASSIFICATION
        # -------------------------------------------------

        if result.boxes.id is not None:
            for box, track_id in zip(
                result.boxes,
                result.boxes.id
            ):
                detection_class_id = int(box.cls[0])

                # Only process human detections
                if detection_class_id != 0:
                    continue

                x1, y1, x2, y2 = map(
                    int,
                    box.xyxy[0]
                )

                x1 = max(0, min(x1, CAM_W - 1))
                x2 = max(0, min(x2, CAM_W))

                y1 = max(0, min(y1, CAM_H - 1))
                y2 = max(0, min(y2, CAM_H))

                person_id = int(track_id)

                current_ids.add(person_id)
                last_seen_time[person_id] = now

                person = frame[y1:y2, x1:x2]

                if person.size == 0:
                    continue

                prediction = classifier(
                    person,
                    verbose=False
                )

                pred = prediction[0]

                class_id = pred.probs.top1
                class_name = pred.names[class_id]

                # Smooth predictions using recent history
                if person_id not in prediction_history:
                    prediction_history[person_id] = deque(
                        maxlen=30
                    )

                prediction_history[person_id].append(
                    class_name
                )

                counter = Counter(
                    prediction_history[person_id]
                )

                smoothed_class, _ = counter.most_common(1)[0]

                # Initialize status
                if person_id not in current_status:
                    current_status[person_id] = smoothed_class
                    status_start_time[person_id] = now

                # Handle status changes
                elif current_status[person_id] != smoothed_class:
                    old_status = current_status[person_id]

                    activity_history.appendleft({
                        "time": time.strftime("%H:%M:%S"),
                        "person_id": person_id,
                        "old_status": old_status,
                        "new_status": smoothed_class
                    })

                    current_status[person_id] = smoothed_class
                    status_start_time[person_id] = now

                duration = int(
                    now - status_start_time[person_id]
                )

                duration_text = format_duration(duration)

                # Determine alert status
                alert_info = get_alert_info(
                    current_status[person_id],
                    duration
                )

                if alert_info:
                    (
                        alert_title,
                        advice1,
                        advice2,
                        advice3,
                        alert_type
                    ) = alert_info

                    is_alert = True

                else:
                    alert_title = ""
                    advice1 = ""
                    advice2 = ""
                    advice3 = ""

                    alert_type = None
                    is_alert = False

                # Save information for the dashboard
                live_people[person_id] = {
                    "status": current_status[person_id],
                    "duration": duration,
                    "alert": is_alert,
                    "alert_type": alert_type,
                    "alert_title": alert_title,
                    "advice1": advice1,
                    "advice2": advice2,
                    "advice3": advice3
                }

                # Draw bounding box
                box_color = RED if is_alert else (0, 200, 0)

                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    box_color,
                    2
                )

                # Draw activity and duration labels
                text1 = f"ID {person_id}: {smoothed_class}"
                text2 = f"Durasi: {duration_text}"

                label_y1 = max(24, y1 - 36)
                label_y2 = max(48, y1 - 10)

                draw_text(
                    text1,
                    (x1, label_y1 + HEADER_H),
                    19,
                    box_color,
                    "bold",
                    stroke=2
                )

                draw_text(
                    text2,
                    (x1, label_y2 + HEADER_H),
                    17,
                    box_color,
                    "bold",
                    stroke=2
                )

                # Draw warning label on the bounding box
                if is_alert:
                    alert_label = (
                        "SITTING TOO LONG"
                        if alert_type == "sitting"
                        else "STANDING TOO LONG"
                    )

                    label_top = min(y2 - 1, y1 + 4)
                    label_bottom = min(y2 - 1, y1 + 32)

                    if label_bottom > label_top:
                        cv2.rectangle(
                            frame,
                            (x1, label_top),
                            (min(x2, x1 + 230), label_bottom),
                            (45, 45, 190),
                            -1
                        )

                        draw_text(
                            alert_label,
                            (
                                x1 + 8,
                                min(y2 - 1, y1 + 24) + HEADER_H
                            ),
                            15,
                            (255, 255, 255),
                            "bold"
                        )

        # -------------------------------------------------
        # REMOVE LOST TRACKING DATA
        # -------------------------------------------------

        for old_id in list(last_seen_time.keys()):
            if now - last_seen_time[old_id] > TRACK_TIMEOUT_SECONDS:
                last_seen_time.pop(old_id, None)
                prediction_history.pop(old_id, None)
                current_status.pop(old_id, None)
                status_start_time.pop(old_id, None)

        # -------------------------------------------------
        # CREATE DASHBOARD CANVAS
        # -------------------------------------------------

        canvas_h = CAM_H + HEADER_H + FOOTER_H
        canvas_w = CAM_W + SIDEBAR_W

        dashboard = np.zeros(
            (canvas_h, canvas_w, 3),
            dtype=np.uint8
        )

        dashboard[:] = BG_NAVY

        dashboard[
            HEADER_H:HEADER_H + CAM_H,
            :CAM_W
        ] = frame

        # Count current people and active alerts
        total_people = len(live_people)

        active_alerts = sum(
            1
            for info in live_people.values()
            if info["alert"]
        )

        # -------------------------------------------------
        # SELECT DASHBOARD PAGE
        # -------------------------------------------------

        if show_history:
            draw_header(
                dashboard,
                "ACTIVITY HISTORY",
                "Riwayat perubahan status aktivitas"
            )

            draw_history_page(dashboard)

        elif show_alerts:
            draw_header(
                dashboard,
                "ALERT MONITOR",
                "Informasi peringatan aktivitas"
            )

            draw_alert_page(
                dashboard,
                live_people
            )

        else:
            draw_header(
                dashboard,
                "ACTIVITY MONITOR",
                "COMPUTER VISION  |  WEBCAM"
            )

            draw_monitoring_page(
                dashboard,
                live_people,
                total_people,
                active_alerts
            )

        # -------------------------------------------------
        # FOOTER
        # -------------------------------------------------

        footer_y = HEADER_H + CAM_H

        cv2.rectangle(
            dashboard,
            (0, footer_y),
            (canvas_w, canvas_h),
            HEADER_NAVY,
            -1
        )

        draw_text(
            "H: Riwayat  |  P: Peringatan  |  Q: Keluar",
            (20, footer_y + 26),
            14,
            TEXT_WHITE,
            "semibold"
        )

        current_time = time.strftime("%H:%M:%S")

        draw_text(
            current_time,
            (canvas_w - 20, footer_y + 26),
            15,
            TEXT_MUTED,
            "semibold",
            anchor="rs"
        )

        # Render all queued text
        dashboard = render_text(dashboard)

        # Show dashboard
        cv2.imshow(
            WINDOW_NAME,
            dashboard
        )

        # -------------------------------------------------
        # KEYBOARD CONTROLS
        # -------------------------------------------------

        key = cv2.waitKey(1) & 0xFF

        if key == ord("h"):
            if show_history:
                # Jika halaman history sedang terbuka,
                # kembali ke halaman monitoring utama
                show_history = False
            else:
                # Buka halaman history
                show_history = True
                show_alerts = False

        elif key == ord("p"):
            if show_alerts:
                # Jika halaman peringatan sedang terbuka,
                # kembali ke halaman monitoring utama
                show_alerts = False
            else:
                # Buka halaman peringatan
                show_alerts = True
                show_history = False

        elif key == ord("q"):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()