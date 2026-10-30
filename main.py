import numpy as np
import cv2
import pygame
import time
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.vision import HandLandmarksConnections

HAND_CONNECTIONS = HandLandmarksConnections.HAND_CONNECTIONS
pygame.mixer.init()

def draw_landmarks(frame, hand_landmarks):
    h, w, _ = frame.shape
    
    # Convert each landmark to pixel coordinates
    points = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks]
    
    # Draw lines between connected landmarks (green)
    for con in HAND_CONNECTIONS:
        cv2.line(frame, points[con.start], points[con.end], (0,255,0), 2)
        
    # Draw a dot at each landmark (red, filled)
    for x, y in points:
        cv2.circle(frame, (x, y), 4, (255,0,0), -1)

base_options = mp_python.BaseOptions(model_asset_path="hand_landmarker.task")

options = mp_vision.HandLandmarkerOptions(
    base_options = base_options,
    running_mode = mp_vision.RunningMode.VIDEO,
    num_hands = 2
)

landmarker = mp_vision.HandLandmarker.create_from_options(options)
sound_mapping = {
    1 : "sounds_swaras/1_Sa.mp3",
    2 : "sounds_swaras/2_Re.mp3",
    3 : "sounds_swaras/3_Ga.mp3",
    4 : "sounds_swaras/4_Ma.mp3",
    5 : "sounds_swaras/5_Pa.mp3",
    6 : "sounds_swaras/6_Dha.mp3",
    7 : "sounds_swaras/7_Ni.mp3",
    8 : "sounds_swaras/8_Sa_upper.mp3"
}

sounds = {
    1 : "Sa",
    2 : "Re",
    3 : "Ga",
    4 : "Ma",
    5 : "Pa",
    6 : "Dha",
    7 : "Ni",
    8 : "SAA"
}

last_count = None
last_played = None
cap = cv2.VideoCapture(0)
before = time.time()
thumbs_up = True

while cap.isOpened():
    ok, frame = cap.read()
    
    if not ok:
        break
    
    #Frame rate calc
    now = time.time()
    fps = 1/(now - before)
    before = now
    
    cv2.flip(frame, 1)
    #Frame processing
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format = mp.ImageFormat.SRGB, data = frame)
    timestamp_ms = int(now * 1000)
    result = landmarker.detect_for_video(mp_image, timestamp_ms)
    
    #Finger count detection, gesture detection and hand drawing code
    finger_count = 0
    if result.hand_landmarks:
        for hand_landmarks, handedness in zip(result.hand_landmarks, result.handedness) :
            draw_landmarks(frame, hand_landmarks)
            
            if hand_landmarks[4].x > hand_landmarks[3].x and handedness[0].category_name == "Right" or hand_landmarks[4].x < hand_landmarks[3].x and handedness[0].category_name == "Left":
                finger_count += 1
                
            for tip in (8, 12, 16, 20):
                if hand_landmarks[tip].y < hand_landmarks[tip - 2].y:
                    finger_count += 1

    #Playing the relevant sound
    if finger_count in sound_mapping and not pygame.mixer.music.get_busy():
        if finger_count != last_count:
            pygame.mixer.music.load(sound_mapping[finger_count])
            pygame.mixer.music.play()
            last_count = finger_count
            last_played = sounds[finger_count]
            frame = 255 * np.ones_like(frame)
                    
    cv2.putText(frame, 
                f"FPS: {fps:.1f}, Hands: {len(result.hand_landmarks)}, Fingers: {finger_count}, Sound : {last_played}", 
                (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
            
    #Display output code
    cv2.imshow("window", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cap.release()   
cv2.destroyAllWindows()
landmarker.close()