import cv2
import mediapipe as mp
import numpy as np
from keras.models import load_model

actions = [i for i in ['clear', 'space']]
seq_length = 30

model = load_model('alphabet/clear_space.keras')

# mediapipe 기본 설정
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

cap = cv2.VideoCapture(0)

# 이거는 나중에 이해
seq = []
# 세 개의 액션이 같으면 그 동작을 출력
action_seq = []

while cap.isOpened():
    ret, frame = cap.read()

    frame = cv2.flip(frame, 1)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(frame)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    # 손 랜드마크 감지
    if result.multi_hand_landmarks is not None:
        for res in result.multi_hand_landmarks:
            # 21개의 손 랜드마크 각각 정보 4개 저장
            joint = np.zeros((21, 4))
            for j, lm in enumerate(res.landmark):
                joint[j] = [lm.x, lm.y, lm.z, lm.visibility]

            v1 = joint[[0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19], :3]
            v2 = joint[[i for i in range(1, 21)], :3]
            v = v2 - v1  # 3차원에서의 거리 구하기 (벡터)

            v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

            angle = np.arccos(
                np.einsum(
                    'nt,nt->n',
                    v[[0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18], :],
                    v[[1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19], :]
                )
            )

            angle = np.degrees(angle)

            d = np.concatenate([joint.flatten(), angle])
            seq.append(d)

            mp_drawing.draw_landmarks(frame, res, mp_hands.HAND_CONNECTIONS)

            if len(seq) < seq_length:
                continue

            input_data = np.expand_dims(np.array(seq[-seq_length:], dtype=np.float32), axis=0)

            # 불러온 모델에 데이터를 넣고 저장된 가중치들을 이용해서 값을 예측한다.
            y_pred = model.predict(input_data).squeeze()

            i_pred = int(np.argmax(y_pred))
            conf = y_pred[i_pred]
            print(conf)
            if conf < 0.8:
                continue

            action = actions[i_pred]
            action_seq.append(action)

            if len(action_seq) < 3:
                continue

            # 여기는 나중에 수정해본다.
            this_action = '???'
            if action_seq[-1] == action_seq[-2] == action_seq[-3]:
                this_action = action

            cv2.putText(
                frame,
                text=f'{this_action.upper()}',
                org=(int(res.landmark[0].x * frame.shape[1]), int(res.landmark[0].y * frame.shape[0] + 20)),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=1,
                color=(255, 255, 255),
                thickness=3
            )

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) == ord('q'):
        break
