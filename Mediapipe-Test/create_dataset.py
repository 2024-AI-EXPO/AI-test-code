import os
import cv2
import time
import numpy as np
import mediapipe as mp

# a, e 구분은 필요할 듯 <- 일단 보류
# g 데이터 조금 더 정확하게 수집 <- 약간 우선 시 해야 함
# j, z 작동 잘 함

actions = [i for i in ['clear', 'space']]  # 원하는 동작 설정
seq_length = 30  # LSTM 때문
secs_for_action = 45  # 학습 시간 (초)

# mediapipe에 있는 hands 모델
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=1,  # 2로 바꿀 예정x
    min_tracking_confidence=0.5,
    min_detection_confidence=0.5
)

cap = cv2.VideoCapture(0)

file = 'dataset'

created_time = int(time.time())
os.makedirs(file, exist_ok=True)  # if 문이 필요가 없음

while cap.isOpened():
    for idx, action in enumerate(actions):
        ret, frame = cap.read()
        data = []

        frame = cv2.flip(frame, 1)  # 이건 왜 있는지 잘 모름
        cv2.putText(
            frame,
            text=f'Waiting for collecting {action.upper()} action',
            org=(10, 30),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=1,
            color=(255, 255, 255),
            thickness=3
        )
        cv2.imshow('frame', frame)
        cv2.waitKey(7000)  # 7초

        start_time = time.time()  # 처음 측정 시간

        while time.time() - start_time < secs_for_action:  # 정한 시간 이하로 측정하기
            ret, frame = cap.read()

            # frame 작업
            frame = cv2.flip(frame, 1)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(frame)  # 아마도 frame에 대한 결과값을 내는 작업
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            # 이거 공부
            if result.multi_hand_landmarks is not None:
                for res in result.multi_hand_landmarks:
                    # 손바닥 랜드마크는 총 21개, 여기에 있는 정보 총 4개를 한번에 저장
                    joint = np.zeros((21, 4))
                    for j, lm in enumerate(res.landmark):
                        joint[j] = [lm.x, lm.y, lm.x, lm.visibility]

                    # 시작
                    # 손가락 각도 계산 (x, y, z를 이용해서 계산)
                    v1 = joint[[0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19], :3]
                    v2 = joint[[i for i in range(1, 21)], :3]
                    v = v2 - v1  # 20행, 3열

                    # 1-norm 벡터의 방향만 알고 싶을 때
                    v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]  # 열 추가

                    # 벡터의 내적(곱하기)를 역삼각함수(arccos) 함수로 구한다.
                    # ********* 행렬 공부 필수 + np 수학 공부를 좀 더.... *********

                    # 15개
                    angle = np.arccos(
                        np.einsum(
                            'nt,nt->n',
                            v[[0, 1, 2, 4, 5, 6, 8, 9, 10,12, 13, 14, 16, 17, 18], :],
                            v[[1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19], :]
                        )
                    )

                    # 라디안이 아닌 평소 쓰는 각도로 변환
                    angle = np.degrees(angle)
                    # 끝

                    # 어떤 제스처인지 구분
                    angle_label = np.array([angle], dtype=np.float32)
                    angle_label = np.append(angle_label, idx)

                    d = np.concatenate([joint.flatten(), angle_label])
                    data.append(d)

                    # 손 위치 그리기 (멍청....)
                    mp_drawing.draw_landmarks(frame, res, mp_hands.HAND_CONNECTIONS)

            cv2.imshow('frame', frame)
            if cv2.waitKey(1) == ord('q'):
                break

        # 시퀀스 데이터로 저장
        data = np.array(data)

        full_seq_data = []
        # LSTM 시퀀스 설정 크기대로 잘라서 저장
        for seq in range(len(data) - seq_length):
            full_seq_data.append(data[seq:seq + seq_length])
        full_seq_data = np.array(full_seq_data)
        print(action, full_seq_data.shape)
        np.save(os.path.join(file, f'seq_{action}'), full_seq_data)
    break
