import cv2
import mediapipe as mp
import numpy as np
import pyttsx3

# Initialize text-to-speech engine
engine = pyttsx3.init()

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Function to calculate the angle between three points
def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    ba = a - b
    bc = c - b

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)
    return np.degrees(angle)

cap = cv2.VideoCapture(0)

curl_count = 0
curl_stage = "down"

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark

            # Get the necessary joint coordinates
            shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                        landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
            elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                     landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
            wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                     landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]

            # Calculate the elbow angle
            elbow_angle = calculate_angle(shoulder, elbow, wrist)

            # Draw joint connections
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            # Display elbow angle on screen
            cv2.putText(image, f'Elbow Angle: {int(elbow_angle)} deg', (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            # Curl detection logic (state transition)
            if elbow_angle > 160:  # Arm fully extended (Down position)
                if curl_stage == "up":
                    engine.say("Lower your arm fully!")
                    engine.runAndWait()
                curl_stage = "down"

            if elbow_angle < 45 and curl_stage == "down":  # Curl completed (Up position)
                curl_stage = "up"
                curl_count += 1  # Increase curl count
                engine.say("Curl Up! Good job!")
                engine.runAndWait()

            # Give feedback on how much to bend
            if 160 > elbow_angle > 45:  # Midway guidance
                remaining_bend = int(elbow_angle - 45)
                cv2.putText(image, f"Bend more: {remaining_bend}° left!", (50, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                if remaining_bend > 20:  # Speak guidance when the arm is still far from goal
                    engine.say(f"Bend more! {remaining_bend} degrees left!")
                    engine.runAndWait()

            # Display curl count
            cv2.putText(image, f'Curls: {curl_count}', (50, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        cv2.imshow("Bicep Curl Tracker", image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
