import cv2
import numpy as np
import mediapipe as mp

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Setup canvas
cap = cv2.VideoCapture(0)
canvas = None
tool = "free_draw"
draw_color = (255, 0, 255)

# Tool selection positions
tool_coords = {
    "free_draw": (50, 50),
    "line": (150, 50),
    "rectangle": (250, 50),
    "circle": (350, 50),
    "erase": (450, 50)
}

def fingers_up(hand_landmarks):
    finger_tips_ids = [8, 12]  # Index, Middle
    fingers = []
    for id in finger_tips_ids:
        tip = hand_landmarks.landmark[id]
        pip = hand_landmarks.landmark[id - 2]
        fingers.append(tip.y < pip.y)
    return fingers

def get_position(landmark, shape):
    h, w = shape
    return int(landmark.x * w), int(landmark.y * h)

prev_point = None

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    h, w, c = frame.shape

    if canvas is None:
        canvas = np.zeros_like(frame)

    # Convert to RGB
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        for hand_landmark in results.multi_hand_landmarks:
            lm_list = hand_landmark.landmark
            index_finger = get_position(lm_list[8], (h, w))
            middle_finger = get_position(lm_list[12], (h, w))
            thumb_finger = get_position(lm_list[4], (h, w))

            up = fingers_up(hand_landmark)

            if up == [True, False]:  # Only index up = tool selection
                for name, pos in tool_coords.items():
                    if abs(index_finger[0] - pos[0]) < 40 and abs(index_finger[1] - pos[1]) < 40:
                        tool = name

            elif up == [True, True]:  # Both up = draw
                if tool == "free_draw":
                    if prev_point is None:
                        prev_point = index_finger
                    cv2.line(canvas, prev_point, index_finger, draw_color, 5)
                    prev_point = index_finger

                elif tool == "line":
                    if prev_point is None:
                        prev_point = index_finger
                    cv2.line(canvas, prev_point, index_finger, draw_color, 2)

                elif tool == "rectangle":
                    if prev_point is None:
                        prev_point = index_finger
                    cv2.rectangle(canvas, prev_point, index_finger, draw_color, 2)

                elif tool == "circle":
                    if prev_point is None:
                        prev_point = index_finger
                    radius = int(((prev_point[0] - index_finger[0])**2 + (prev_point[1] - index_finger[1])**2)**0.5)
                    cv2.circle(canvas, prev_point, radius, draw_color, 2)

                elif tool == "erase":
                    cv2.circle(canvas, index_finger, 30, (0, 0, 0), -1)

            else:
                prev_point = None

            mp_draw.draw_landmarks(frame, hand_landmark, mp_hands.HAND_CONNECTIONS)

    # Show tool buttons
    for name, pos in tool_coords.items():
        color = (0, 255, 0) if name == tool else (255, 255, 255)
        cv2.rectangle(frame, (pos[0] - 30, pos[1] - 30), (pos[0] + 30, pos[1] + 30), color, 2)
        cv2.putText(frame, name, (pos[0] - 30, pos[1] - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Merge canvas with video
    gray_canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray_canvas, 20, 255, cv2.THRESH_BINARY)
    inv_mask = cv2.bitwise_not(mask)
    frame_bg = cv2.bitwise_and(frame, frame, mask=inv_mask)
    draw_fg = cv2.bitwise_and(canvas, canvas, mask=mask)
    combined = cv2.add(frame_bg, draw_fg)

    cv2.imshow("Virtual Drawing", combined)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

 
 
