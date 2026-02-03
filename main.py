import cv2
import mediapipe as mp
import numpy as np
import os

IMAGE_PATHS = {
    "THUMBS_UP": "thumbs_up.jpg",
    "POINTING": "pointing.jpg",
    "NEUTRAL": "neutral.jpg",
    "THINKING": "thinking.jpg"
}

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
mp_face_mesh = mp.solutions.face_mesh

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.6
)

face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.6
)

cap = cv2.VideoCapture(0)
print("Camera object:", cap.isOpened())


while cap.isOpened():
    ret, frame = cap.read()
    print("Frame read:", ret)
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(rgb)

    gesture = "NEUTRAL"

    if results.multi_hand_landmarks:
        gesture = "THUMBS_UP"
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
            )

    img_path = IMAGE_PATHS.get(gesture)
    if img_path and os.path.exists(img_path):
        img = cv2.imread(img_path)
        img = cv2.resize(img, (300, 300))
        frame[0:300, 0:300] = img

    cv2.putText(
        frame,
        f"Gesture: {gesture}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.imshow("Monkey Meme", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
