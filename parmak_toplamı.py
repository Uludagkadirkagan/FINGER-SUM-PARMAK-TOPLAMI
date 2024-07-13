import cv2
import mediapipe as mp
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

cap = cv2.VideoCapture(0)

with mp_hands.Hands(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            break

        height, width, _ = image.shape
        height_limit = height * 0.5  # Yükseklik limitini ayarla

        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = hands.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        total_fingers = 0
        if results.multi_hand_landmarks:
            for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
                mp_drawing.draw_landmarks(
                    image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Elin merkezini hesapla
                coords = np.zeros((21, 2))
                for j, lm in enumerate(hand_landmarks.landmark):
                    coords[j] = [lm.x * width, lm.y * height]
                center = coords.mean(axis=0).astype(int)

                finger_up = [False, False, False, False, False]
                if (hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y <
                        hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP].y and
                        hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y < height_limit):
                    finger_up[0] = True
                if (hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y <
                        hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].y and
                        hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y < height_limit):
                    finger_up[1] = True
                if (hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y <
                        hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y and
                        hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y < height_limit):
                    finger_up[2] = True
                if (hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].y <
                        hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP].y and
                        hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].y < height_limit):
                    finger_up[3] = True
                if (hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].y <
                        hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP].y and
                        hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].y < height_limit):
                    finger_up[4] = True

                count_fingers = sum(finger_up)
                total_fingers += count_fingers

                # Kalkık parmak sayısını görüntü üzerinde yazdırma
                cv2.putText(image, str(count_fingers), (center[0], center[1] - 50), cv2.FONT_HERSHEY_SIMPLEX, 3,
                            (255, 0, 0), 5)

            # Toplam parmak sayısını yazdır
            cv2.putText(image, "Toplam: " + str(total_fingers), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        cv2.imshow('MediaPipe Hands', image)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()