import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import cv2
import csv
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision.models.detection import keypointrcnn_resnet50_fpn
from torchvision.models.detection import KeypointRCNN_ResNet50_FPN_Weights


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

VIDEO_PATH = os.path.join(BASE_DIR, "7556227-uhd_3840_2160_25fps.mp4")
OUTPUT_PATH = os.path.join(BASE_DIR, "output_fast.mp4")
CSV_PATH = os.path.join(BASE_DIR, "motion_log.csv")
GRAPH_PATH = os.path.join(BASE_DIR, "motion_graph.png")
REPORT_PATH = os.path.join(BASE_DIR, "analiz_raporu.txt")
DASHBOARD_PATH = os.path.join(BASE_DIR, "dashboard.png")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DETECTION_THRESHOLD = 0.6
KEYPOINT_THRESHOLD = 0.0

MOTION_THRESHOLD = 4.0
TURN_THRESHOLD = 15.0
IMMOBILE_SECONDS = 5

SUDDEN_MOTION_THRESHOLD = 25.0
HIGH_CENTER_SHIFT_THRESHOLD = 35.0

TURN_X_THRESHOLD = 20
UPWARD_Y_THRESHOLD = 15

RISK_MOTION_WEIGHT = 0.4
RISK_CENTER_WEIGHT = 0.4
RISK_IMMOBILE_WEIGHT = 0.2

RESIZE_WIDTH = 480
RESIZE_HEIGHT = 270
SKIP_FRAMES = 10
SMOOTH_ALPHA = 0.6

TEXT_SCALE = 0.60
TEXT_THICKNESS = 2

UPPER_BODY_POINTS = [0, 5, 6, 7, 8, 9, 10]

SKELETON = [
    (5, 6),
    (5, 7), (7, 9),
    (6, 8), (8, 10)
]

motion_values = []
center_values = []
state_values = []
time_values = []
alarm_values = []
risk_values = []
turn_direction_values = []


def load_model():
    weights = KeypointRCNN_ResNet50_FPN_Weights.DEFAULT
    model = keypointrcnn_resnet50_fpn(weights=weights)
    model.eval()
    model.to(DEVICE)
    return model


def preprocess_frame(frame_bgr):
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    frame_tensor = torch.from_numpy(frame_rgb).float() / 255.0
    frame_tensor = frame_tensor.permute(2, 0, 1)
    return frame_tensor.to(DEVICE)


def draw_pose(frame, keypoints):
    for i in UPPER_BODY_POINTS:
        x, y, v = keypoints[i]
        if v > KEYPOINT_THRESHOLD:
            cv2.circle(frame, (int(x), int(y)), 3, (0, 255, 0), -1)

    for a, b in SKELETON:
        x1, y1, v1 = keypoints[a]
        x2, y2, v2 = keypoints[b]

        if v1 > KEYPOINT_THRESHOLD and v2 > KEYPOINT_THRESHOLD:
            cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)


def compute_motion(prev_keypoints, curr_keypoints):
    distances = []

    for i in UPPER_BODY_POINTS:
        x1, y1, v1 = prev_keypoints[i]
        x2, y2, v2 = curr_keypoints[i]

        if v1 > KEYPOINT_THRESHOLD and v2 > KEYPOINT_THRESHOLD:
            d = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            if d < 50:
                distances.append(d)

    if len(distances) == 0:
        return 0.0

    return float(np.mean(distances))


def compute_body_center(keypoints):
    valid_points = []

    for i in UPPER_BODY_POINTS:
        x, y, v = keypoints[i]
        if v > KEYPOINT_THRESHOLD:
            valid_points.append([x, y])

    if len(valid_points) == 0:
        return None

    valid_points = np.array(valid_points)
    return np.mean(valid_points, axis=0)


def smooth_value(previous_value, current_value, alpha=SMOOTH_ALPHA):
    if previous_value is None:
        return current_value
    return alpha * current_value + (1 - alpha) * previous_value


def classify_state(motion_score, center_shift, immobile_time):
    if immobile_time >= IMMOBILE_SECONDS:
        return "HAREKETSIZLIK_UYARISI"
    elif center_shift > TURN_THRESHOLD and motion_score > MOTION_THRESHOLD:
        return "POZISYON_DEGISTIRME / DONME"
    elif motion_score > MOTION_THRESHOLD:
        return "AKTIF_HAREKET"
    else:
        return "MINOR_HAREKET"


def detect_alarm(motion_score, center_shift, immobile_time):
    if immobile_time >= IMMOBILE_SECONDS:
        return "BASI_YARASI_RISKI"
    elif motion_score >= SUDDEN_MOTION_THRESHOLD:
        return "ANI_HAREKET_UYARISI"
    elif center_shift >= HIGH_CENTER_SHIFT_THRESHOLD:
        return "BELIRGIN_POZISYON_DEGISIMI"
    else:
        return "YOK"


def detect_turn_direction(prev_center, curr_center):
    if prev_center is None or curr_center is None:
        return "BELIRSIZ"

    dx = curr_center[0] - prev_center[0]
    dy = prev_center[1] - curr_center[1]

    if dx > TURN_X_THRESHOLD:
        return "SAGA_KAYMA"
    elif dx < -TURN_X_THRESHOLD:
        return "SOLA_KAYMA"
    elif dy > UPWARD_Y_THRESHOLD:
        return "DOGRULMA"
    else:
        return "NORMAL"


def calculate_risk_score(motion_score, center_shift, immobile_time):
    motion_risk = min((motion_score / SUDDEN_MOTION_THRESHOLD) * 100, 100)
    center_risk = min((center_shift / HIGH_CENTER_SHIFT_THRESHOLD) * 100, 100)
    immobile_risk = min((immobile_time / IMMOBILE_SECONDS) * 100, 100)

    risk_score = (
        motion_risk * RISK_MOTION_WEIGHT +
        center_risk * RISK_CENTER_WEIGHT +
        immobile_risk * RISK_IMMOBILE_WEIGHT
    )

    return int(risk_score)


def get_status_color(risk_score, alarm):
    if alarm != "YOK" or risk_score >= 70:
        return (0, 0, 255)
    elif risk_score >= 40:
        return (0, 165, 255)
    else:
        return (0, 255, 0)


def create_dashboard(
    total_time,
    avg_motion,
    max_motion,
    avg_center,
    max_center,
    avg_risk,
    max_risk,
    position_change_event_count,
    active_motion_event_count,
    minor_motion_event_count,
    immobile_warning_event_count,
    alarm_event_count,
    sudden_alarm_count,
    position_alarm_count,
    pressure_alarm_count,
    right_shift_count,
    left_shift_count,
    sitting_up_count
):
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.axis("off")

    ax.text(
        0.5,
        0.96,
        "HASTA YATAGI HAREKET ANALIZ DASHBOARD",
        ha="center",
        va="center",
        fontsize=22,
        fontweight="bold"
    )

    left_text = f"""
GENEL OZET

Toplam Analiz Suresi: {total_time:.2f} saniye

Ortalama Risk Score: %{avg_risk:.2f}
Maksimum Risk Score: %{max_risk:.2f}

HAREKET ANALIZI

Ortalama Motion Score: {avg_motion:.2f}
Maksimum Motion Score: {max_motion:.2f}

Ortalama Center Shift: {avg_center:.2f}
Maksimum Center Shift: {max_center:.2f}
"""

    middle_text = f"""
OLAY BAZLI SAYIM

Pozisyon Degistirme / Donme: {position_change_event_count}

Aktif Hareket: {active_motion_event_count}

Minor Hareket: {minor_motion_event_count}

Hareketsizlik Uyarisi: {immobile_warning_event_count}

YON / KAYMA ANALIZI

Saga Kayma: {right_shift_count}

Sola Kayma: {left_shift_count}

Dogrulma: {sitting_up_count}
"""

    right_text = f"""
ALARM OZETI

Toplam Alarm: {alarm_event_count}

Ani Hareket Alarmi: {sudden_alarm_count}

Belirgin Pozisyon Degisimi: {position_alarm_count}

Basi Yarasi Riski: {pressure_alarm_count}

ESIKLER

Motion Threshold: {MOTION_THRESHOLD}

Turn Threshold: {TURN_THRESHOLD}

Sudden Motion: {SUDDEN_MOTION_THRESHOLD}

High Center Shift: {HIGH_CENTER_SHIFT_THRESHOLD}
"""

    ax.text(0.04, 0.86, left_text, ha="left", va="top", fontsize=12, family="monospace")
    ax.text(0.37, 0.86, middle_text, ha="left", va="top", fontsize=12, family="monospace")
    ax.text(0.69, 0.86, right_text, ha="left", va="top", fontsize=12, family="monospace")

    if max_risk >= 70:
        yorum = (
            "GENEL YORUM: Analiz boyunca belirgin pozisyon "
            "degisimleri ve yuksek riskli hareketler tespit edildi. "
            "Uzun sureli hareketsizlik gozlenmedi."
        )
    elif avg_risk >= 40:
        yorum = (
            "GENEL YORUM: Orta duzey hareketlilik ve "
            "duzenli pozisyon degisimleri gozlemlendi."
        )
    else:
        yorum = (
            "GENEL YORUM: Dusuk riskli ve daha stabil "
            "hareket profili izlendi."
        )

    ax.text(
        0.04,
        0.10,
        yorum,
        ha="left",
        va="center",
        fontsize=14,
        fontweight="bold"
    )

    ax.text(
        0.04,
        0.04,
        "Risk Score Aciklamasi: Risk Score; hareket siddeti, pozisyon degisimi "
        "ve hareketsizlik suresine bagli davranissal risk seviyesini temsil eder.",
        ha="left",
        va="center",
        fontsize=11
    )

    plt.subplots_adjust(left=0.03, right=0.97, top=0.95, bottom=0.05)
    plt.savefig(DASHBOARD_PATH, dpi=200, bbox_inches="tight")
    plt.close()


def main():
    print(f"Cihaz: {DEVICE}")

    model = load_model()
    cap = cv2.VideoCapture(VIDEO_PATH)

    if not cap.isOpened():
        print("Video acilamadi.")
        return

    original_fps = cap.get(cv2.CAP_PROP_FPS)

    if original_fps <= 0:
        original_fps = 25.0

    output_fps = original_fps
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    writer = cv2.VideoWriter(
        OUTPUT_PATH,
        fourcc,
        output_fps,
        (RESIZE_WIDTH, RESIZE_HEIGHT)
    )

    csv_file = open(CSV_PATH, mode="w", newline="", encoding="utf-8")
    csv_writer = csv.writer(csv_file)

    csv_writer.writerow([
        "time_sec",
        "frame",
        "raw_motion_score",
        "filtered_motion_score",
        "raw_center_shift",
        "filtered_center_shift",
        "state",
        "immobile_time",
        "alarm",
        "risk_score",
        "turn_direction"
    ])

    prev_keypoints = None
    prev_center = None

    immobile_frames = 0
    frame_idx = 0

    last_box = None
    last_keypoints = None

    last_motion_score = 0.0
    last_center_shift = 0.0
    last_state = "BEKLENIYOR"
    last_immobile_time = 0.0
    last_alarm = "YOK"
    last_risk_score = 0
    last_turn_direction = "NORMAL"

    previous_filtered_motion = None
    previous_filtered_center = None

    previous_event_state = None
    previous_alarm = "YOK"

    position_change_event_count = 0
    active_motion_event_count = 0
    minor_motion_event_count = 0
    immobile_warning_event_count = 0
    alarm_event_count = 0

    with torch.no_grad():
        while True:
            ret, frame = cap.read()

            if not ret:
                break

            frame_idx += 1
            frame = cv2.resize(frame, (RESIZE_WIDTH, RESIZE_HEIGHT))

            run_model = (frame_idx % SKIP_FRAMES == 0)

            if run_model:
                input_tensor = preprocess_frame(frame)
                outputs = model([input_tensor])[0]

                if len(outputs["scores"]) > 0:
                    scores = outputs["scores"].detach().cpu().numpy()
                    best_idx = int(np.argmax(scores))

                    if scores[best_idx] >= DETECTION_THRESHOLD:
                        box = outputs["boxes"][best_idx].detach().cpu().numpy().astype(int)
                        keypoints = outputs["keypoints"][best_idx].detach().cpu().numpy()

                        raw_motion_score = 0.0
                        raw_center_shift = 0.0
                        curr_center = compute_body_center(keypoints)

                        if prev_keypoints is not None:
                            raw_motion_score = compute_motion(prev_keypoints, keypoints)

                        if prev_center is not None and curr_center is not None:
                            raw_center_shift = float(np.linalg.norm(curr_center - prev_center))

                        turn_direction = detect_turn_direction(prev_center, curr_center)

                        filtered_motion_score = smooth_value(previous_filtered_motion, raw_motion_score)
                        filtered_center_shift = smooth_value(previous_filtered_center, raw_center_shift)

                        previous_filtered_motion = filtered_motion_score
                        previous_filtered_center = filtered_center_shift

                        if filtered_motion_score < MOTION_THRESHOLD:
                            immobile_frames += SKIP_FRAMES
                        else:
                            immobile_frames = 0

                        immobile_time = immobile_frames / original_fps

                        state = classify_state(filtered_motion_score, filtered_center_shift, immobile_time)
                        alarm = detect_alarm(filtered_motion_score, filtered_center_shift, immobile_time)
                        risk_score = calculate_risk_score(filtered_motion_score, filtered_center_shift, immobile_time)

                        if state != previous_event_state:
                            if state == "POZISYON_DEGISTIRME / DONME":
                                position_change_event_count += 1
                            elif state == "AKTIF_HAREKET":
                                active_motion_event_count += 1
                            elif state == "MINOR_HAREKET":
                                minor_motion_event_count += 1
                            elif state == "HAREKETSIZLIK_UYARISI":
                                immobile_warning_event_count += 1

                            previous_event_state = state

                        if alarm != "YOK" and alarm != previous_alarm:
                            alarm_event_count += 1

                        previous_alarm = alarm
                        time_sec = frame_idx / original_fps

                        motion_values.append(filtered_motion_score)
                        center_values.append(filtered_center_shift)
                        state_values.append(state)
                        time_values.append(time_sec)
                        alarm_values.append(alarm)
                        risk_values.append(risk_score)
                        turn_direction_values.append(turn_direction)

                        csv_writer.writerow([
                            round(time_sec, 2),
                            frame_idx,
                            round(raw_motion_score, 2),
                            round(filtered_motion_score, 2),
                            round(raw_center_shift, 2),
                            round(filtered_center_shift, 2),
                            state,
                            round(immobile_time, 2),
                            alarm,
                            risk_score,
                            turn_direction
                        ])

                        prev_keypoints = keypoints.copy()
                        prev_center = curr_center.copy() if curr_center is not None else None

                        last_box = box
                        last_keypoints = keypoints
                        last_motion_score = filtered_motion_score
                        last_center_shift = filtered_center_shift
                        last_state = state
                        last_immobile_time = immobile_time
                        last_alarm = alarm
                        last_risk_score = risk_score
                        last_turn_direction = turn_direction

            status_color = get_status_color(last_risk_score, last_alarm)

            if last_box is not None:
                x1, y1, x2, y2 = last_box
                cv2.rectangle(frame, (x1, y1), (x2, y2), status_color, 3)

            if last_keypoints is not None:
                draw_pose(frame, last_keypoints)

            cv2.putText(frame, f"Device: {DEVICE}", (15, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, TEXT_SCALE, (0, 255, 255), TEXT_THICKNESS)
            cv2.putText(frame, f"Motion Score: {last_motion_score:.2f}", (15, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, TEXT_SCALE, (255, 255, 255), TEXT_THICKNESS)
            cv2.putText(frame, f"Center Shift: {last_center_shift:.2f}", (15, 75),
                        cv2.FONT_HERSHEY_SIMPLEX, TEXT_SCALE, (255, 255, 255), TEXT_THICKNESS)
            cv2.putText(frame, f"State: {last_state}", (15, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, TEXT_SCALE, (0, 255, 0), TEXT_THICKNESS)
            cv2.putText(frame, f"Immobile Time: {last_immobile_time:.1f}s", (15, 125),
                        cv2.FONT_HERSHEY_SIMPLEX, TEXT_SCALE, (0, 100, 255), TEXT_THICKNESS)
            cv2.putText(frame, f"Alarm: {last_alarm}", (15, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, TEXT_SCALE, status_color, TEXT_THICKNESS)
            cv2.putText(frame, f"Risk Score: %{last_risk_score}", (15, 175),
                        cv2.FONT_HERSHEY_SIMPLEX, TEXT_SCALE, status_color, TEXT_THICKNESS)
            cv2.putText(frame, f"Direction: {last_turn_direction}", (15, 200),
                        cv2.FONT_HERSHEY_SIMPLEX, TEXT_SCALE, (255, 255, 0), TEXT_THICKNESS)

            cv2.rectangle(
                frame,
                (0, 0),
                (RESIZE_WIDTH - 1, RESIZE_HEIGHT - 1),
                status_color,
                3
            )

            writer.write(frame)

    cap.release()
    writer.release()
    csv_file.close()
    cv2.destroyAllWindows()

    print(f"Islem tamamlandi. Kaydedilen video: {OUTPUT_PATH}")

    if len(time_values) > 0:
        plt.figure(figsize=(10, 5))

        plt.plot(time_values, motion_values, label="Filtered Motion Score")
        plt.plot(time_values, center_values, label="Filtered Center Shift")

        plt.axhline(y=MOTION_THRESHOLD, color="green", linestyle="--", linewidth=2, label="Motion Threshold")
        plt.axhline(y=TURN_THRESHOLD, color="orange", linestyle="--", linewidth=2, label="Turn Threshold")
        plt.axhline(y=SUDDEN_MOTION_THRESHOLD, color="red", linestyle="--", linewidth=2, label="Sudden Motion Threshold")
        plt.axhline(y=HIGH_CENTER_SHIFT_THRESHOLD, color="purple", linestyle="--", linewidth=2, label="High Center Shift Threshold")

        plt.xlabel("Zaman (saniye)")
        plt.ylabel("Deger")
        plt.title("Hasta Yatagi Hareket Analizi")
        plt.legend()
        plt.grid(True)
        plt.savefig(GRAPH_PATH)
        plt.close()

        total_time = max(time_values)

        avg_motion = np.mean(motion_values)
        max_motion = np.max(motion_values)

        avg_center = np.mean(center_values)
        max_center = np.max(center_values)

        avg_risk = np.mean(risk_values)
        max_risk = np.max(risk_values)

        sudden_alarm_count = alarm_values.count("ANI_HAREKET_UYARISI")
        position_alarm_count = alarm_values.count("BELIRGIN_POZISYON_DEGISIMI")
        pressure_alarm_count = alarm_values.count("BASI_YARASI_RISKI")

        right_shift_count = turn_direction_values.count("SAGA_KAYMA")
        left_shift_count = turn_direction_values.count("SOLA_KAYMA")
        sitting_up_count = turn_direction_values.count("DOGRULMA")

        create_dashboard(
            total_time,
            avg_motion,
            max_motion,
            avg_center,
            max_center,
            avg_risk,
            max_risk,
            position_change_event_count,
            active_motion_event_count,
            minor_motion_event_count,
            immobile_warning_event_count,
            alarm_event_count,
            sudden_alarm_count,
            position_alarm_count,
            pressure_alarm_count,
            right_shift_count,
            left_shift_count,
            sitting_up_count
        )

        report_text = f"""
--- HASTA YATAGI HAREKET ANALIZ RAPORU ---

Toplam analiz suresi: {total_time:.2f} saniye

Ortalama Motion Score: {avg_motion:.2f}
Maksimum Motion Score: {max_motion:.2f}

Ortalama Center Shift: {avg_center:.2f}
Maksimum Center Shift: {max_center:.2f}

Ortalama Risk Score: %{avg_risk:.2f}
Maksimum Risk Score: %{max_risk:.2f}

Olay Bazli Sayim:
Pozisyon degistirme / donme olayi: {position_change_event_count}
Aktif hareket olayi: {active_motion_event_count}
Minor hareket olayi: {minor_motion_event_count}
Hareketsizlik uyarisi olayi: {immobile_warning_event_count}

Yon / Kayma Analizi:
Saga kayma sayisi: {right_shift_count}
Sola kayma sayisi: {left_shift_count}
Dogrulma sayisi: {sitting_up_count}

Alarm Ozetleri:
Toplam alarm olayi: {alarm_event_count}
Ani hareket alarmi: {sudden_alarm_count}
Belirgin pozisyon degisimi alarmi: {position_alarm_count}
Basi yarasi riski alarmi: {pressure_alarm_count}

Kullanilan Esikler:
Motion Threshold: {MOTION_THRESHOLD}
Turn Threshold: {TURN_THRESHOLD}
Sudden Motion Threshold: {SUDDEN_MOTION_THRESHOLD}
High Center Shift Threshold: {HIGH_CENTER_SHIFT_THRESHOLD}
Turn X Threshold: {TURN_X_THRESHOLD}
Upward Y Threshold: {UPWARD_Y_THRESHOLD}
Immobile Seconds: {IMMOBILE_SECONDS}

Yorum:
Bu analizde ust vucut keypointleri kullanilarak hastanin hareket yogunlugu, pozisyon degisimi, yonlu kayma ve risk durumu incelenmistir.
Motion Score eklem noktalarinin kareler arasi ortalama hareketini, Center Shift ise ust vucut merkezinin yer degisimini temsil eder.
Risk Score; hareket siddeti, merkez kaymasi ve hareketsizlik suresinin agirlikli birlesimi ile hesaplanmistir.

Risk Score Aciklamasi:
Risk Score; hastanin hareket yogunlugu, pozisyon degisimi ve hareketsizlik durumuna bagli davranissal risk seviyesini temsil etmektedir.
"""

        with open(REPORT_PATH, "w", encoding="utf-8") as report_file:
            report_file.write(report_text)

        print("\n--- ANALIZ RAPORU ---")
        print(f"Toplam analiz suresi: {total_time:.2f} saniye")
        print(f"Ortalama Motion Score: {avg_motion:.2f}")
        print(f"Maksimum Motion Score: {max_motion:.2f}")
        print(f"Ortalama Center Shift: {avg_center:.2f}")
        print(f"Maksimum Center Shift: {max_center:.2f}")
        print(f"Ortalama Risk Score: %{avg_risk:.2f}")
        print(f"Maksimum Risk Score: %{max_risk:.2f}")
        print(f"Pozisyon degistirme / donme olayi: {position_change_event_count}")
        print(f"Aktif hareket olayi: {active_motion_event_count}")
        print(f"Minor hareket olayi: {minor_motion_event_count}")
        print(f"Hareketsizlik uyarisi olayi: {immobile_warning_event_count}")
        print(f"Saga kayma sayisi: {right_shift_count}")
        print(f"Sola kayma sayisi: {left_shift_count}")
        print(f"Dogrulma sayisi: {sitting_up_count}")
        print(f"Toplam alarm olayi: {alarm_event_count}")
        print(f"Ani hareket alarmi: {sudden_alarm_count}")
        print(f"Belirgin pozisyon degisimi alarmi: {position_alarm_count}")
        print(f"Basi yarasi riski alarmi: {pressure_alarm_count}")
        print(f"CSV kaydedildi: {CSV_PATH}")
        print(f"Grafik kaydedildi: {GRAPH_PATH}")
        print(f"TXT rapor kaydedildi: {REPORT_PATH}")
        print(f"Dashboard kaydedildi: {DASHBOARD_PATH}")

    if os.path.exists(OUTPUT_PATH):
        os.startfile(OUTPUT_PATH)


if __name__ == "__main__":
    main()