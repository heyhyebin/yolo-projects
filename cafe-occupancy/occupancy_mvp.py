from ultralytics import YOLO
import cv2

model = YOLO("yolov8n.pt")   # 자동 다운로드됨(처음 1번)

SEATS = 20  # 너가 임의로 좌석 수 입력

def occupancy_state(people, seats):
    if seats <= 0:
        return "INVALID"
    ratio = people / seats
    if ratio <= 0.4:
        return "LOW"
    elif ratio <= 0.7:
        return "MEDIUM"
    else:
        return "CROWDED"

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # 윈도우에서 안정적
if not cap.isOpened():
    raise RuntimeError("웹캠을 열 수 없습니다. 다른 앱이 카메라를 쓰는지 확인하세요.")

print("ESC로 종료")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, conf=0.35, classes=[0], verbose=False)[0]

    people = 0
    if results.boxes is not None:
        for box in results.boxes:
            cls_id = int(box.cls[0])
            if model.names.get(cls_id) == "person":
                people += 1

    ratio = (people / SEATS) if SEATS > 0 else 0.0
    state = occupancy_state(people, SEATS)

    annotated = results.plot()
    cv2.putText(annotated, f"Seats: {SEATS}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(annotated, f"People: {people}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(annotated, f"Occ: {ratio:.2f}  State: {state}", (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    cv2.imshow("Occupancy MVP", annotated)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()
