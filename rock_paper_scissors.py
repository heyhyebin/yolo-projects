from ultralytics import YOLO
import cv2
from picamera2 import Picamera2
import time
from gpiozero import Button, TonalBuzzer, LED

pizeo = TonalBuzzer(21)

notes = {
    1: 329.63,
    2: 349.23,
    3: 392.00,
    4: 440.00,
    5: 493.88,
    6: 523.25,
    7: 587.33,
    8: 622.25,
    9: 659.25,
    10: 698.46,
    11: 783.99,
    12: 880.00,
    13: 987.77,
    }

melody = [9,9,9,7,9,11,3,7,3,1,4,6,5,4,3,9,11,12,10,11,9,7,8,6]
    
def mario():
    for n in melody:
        pizeo.play(notes[n])
        time.sleep(0.2)

reset_button = Button(14)
judge_button = Button(15)

# ======== LED 설정 ========
# 왼쪽 플레이어: GPIO 18, 23, 24
right_leds = [LED(18), LED(23), LED(24)]

# 오른쪽 플레이어: GPIO 1rps1234
left_leds = [LED(1), LED(8), LED(7)]

def update_leds(score, leds):
    """
    score: 0~3
    leds: LED 객체 리스트 (길이 3)
    점수만큼 앞에서부터 켜고 나머지는 끈다.
    """
    for i, led in enumerate(leds):
        if i < score:
            led.on()
        else:
            led.off()

# 처음에는 모두 끔
update_leds(0, left_leds)
update_leds(0, right_leds)


# ========= 가위바위보 설정 =========
CLASS_ID_TO_NAME = {
    0: 'rock',      # 바위
    1: 'paper',     # 보
    2: 'scissors',  # 가위
}

# ========= 승패 판정함수 =========
def rps_winner(left_cls, right_cls):
    """
    left_cls, right_cls : int (class id)
    return: -1 = 무승부, 0 = 왼쪽 승, 1 = 오른쪽 승
    """
    if left_cls == right_cls:
        return -1  # draw

    # 바위(0) > 가위(2), 가위(2) > 보(1), 보(1) > 바위(0)
    if (left_cls == 0 and right_cls == 2) or \
       (left_cls == 2 and right_cls == 1) or \
       (left_cls == 1 and right_cls == 0):
        return 0  # left wins
    else:
        return 1  # right wins


# ========= 카메라 준비 =========
picam2 = Picamera2()
config = picam2.create_preview_configuration(
    main={"format": "RGB888", "size": (640, 640)}
)
picam2.configure(config)
picam2.start()

# ========= YOLO 모델 로드 =========
model = YOLO('best.onnx')

# ========= 점수 변수 =========
left_score = 0
right_score = 0

print("ESC: 종료 / 버튼: 승패 판정 / reset 버튼: 점수 리셋 (삼세판)")

while True:
    frame = picam2.capture_array()   # (640, 640, 3)

    results = model(frame, conf=0.2)[0]

    # YOLO에서 그려준 이미지 (bbox가 그려진 프레임)
    annotated = results.plot()

    # ======== detection에서 왼쪽/오른쪽 손 찾기 ========
    hands = []  # [{ "cx": float, "cls_id": int }, ...]

    if results.boxes is not None:
        for i, box in enumerate(results.boxes):
            if i >= 2:
                break
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])

            # 신뢰도 필터링
            if conf < 0.5:
                continue

            # 가위/바위/보 클래스만 사용
            if cls_id not in CLASS_ID_TO_NAME:
                continue

            x1, y1, x2, y2 = box.xyxy[0]
            cx = float((x1 + x2) / 2.0)  # 중심 x

            hands.append({
                "cx": cx,
                "cls_id": cls_id,
            })

    left_hand_cls = None
    right_hand_cls = None
    left_hand_name = None
    right_hand_name = None
    
    if reset_button.is_pressed:
        left_score = 0
        right_score = 0
        update_leds(left_score, left_leds)
        update_leds(right_score, right_leds)
        print("점수 리셋: 0 : 0 (LED 모두 OFF)")
        time.sleep(2)

    # 중심 x 기준으로 정렬해서 왼쪽/오른쪽 결정
    elif len(hands) >= 2:
        hands_sorted = sorted(hands, key=lambda h: h["cx"])
        left_hand_cls = hands_sorted[0]["cls_id"]
        right_hand_cls = hands_sorted[1]["cls_id"]

        left_hand_name = CLASS_ID_TO_NAME[left_hand_cls]
        right_hand_name = CLASS_ID_TO_NAME[right_hand_cls]

        

        # ========= 버튼 누를 때 승패 판정 =========
        if judge_button.is_pressed:
            if left_hand_cls is not None and right_hand_cls is not None:
                result = rps_winner(left_hand_cls, right_hand_cls)
                if result == -1:
                    print("무승부!")
                elif result == 0:
                    left_score += 1
                    print("왼쪽 플레이어 승! ->", left_score, ":", right_score)
                else:
                    right_score += 1
                    print("오른쪽 플레이어 승! ->", left_score, ":", right_score)

                # 점수에 따라 LED 업데이트
                update_leds(left_score, left_leds)
                update_leds(right_score, right_leds)

                # ===== 삼세판 승자 체크 =====
                if left_score == 3:
                    print("왼쪽 플레이어가 3점을 먼저 달성했습니다! 게임 종료")
                    mario()
                    time.sleep(3)
                    break
                elif right_score == 3:
                    print("오른쪽 플레이어가 3점을 먼저 달성했습니다! 게임 종료")
                    mario()
                    time.sleep(3)
                    break
            else:
                print("양손이 모두 인식되지 않았습니다.")
                
            time.sleep(1)
        
    else:
        cv2.putText(annotated, "Need 2 hands detected", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    # ======== 점수 & 안내 텍스트 오버레이 ========
    score_text = f"Score  L:{left_score}  -  R:{right_score}"
    cv2.putText(annotated, score_text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
    cv2.putText(annotated, "Button1: Reset, Button2: Judge, Reset btn: reset  ESC: quit", (10, 620),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    cv2.imshow("YOLO RPS", annotated)   ### 화면 출력 ###

    key = cv2.waitKey(1) & 0xFF

    if key == 27:  # ESC
        break

# 루프 종료 후 정리
picam2.stop()
cv2.destroyAllWindows()
