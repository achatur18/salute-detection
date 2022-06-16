import cv2
import mediapipe as mp
import math
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic

mp_face_mesh = mp.solutions.face_mesh
mp_hands = mp.solutions.hands

def slope(x1, y1, x2, y2): # Line slope given two points:
    return (y2-y1)/(x2-x1)

def angle(s1, s2): 
    return math.degrees(math.atan((s2-s1)/(1+(s2*s1))))

def slope_(lm, x,y,z):
    x=lm[x]
    y=lm[y]
    z=lm[z]
    m1=slope(x.x, x.y, y.x, y.y)
    m2=slope(z.x, z.y, y.x, y.y)

    ang= angle(m1, m2)
    if ang<0:
        ang=ang+180
    return ang

def l2_dist(p1, p2, h, w, c):
    # print(p1.x*w, p1.y*h)
    dx=(p1.x*w-p2.x*w)**2
    dy=(p1.y*h-p2.y*h)**2
    dz=(p1.y*c-p2.y*c)**2
    return (dx+dy+dz)**0.5
# For webcam input:
cap = cv2.VideoCapture(0)
with mp_holistic.Holistic(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as holistic:
  while cap.isOpened():
    success, image = cap.read()
    h, w, c = image.shape
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue

    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = holistic.process(image)

    # Draw landmark annotation on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

#12, 14, 16
#11, 13, 15
    if (results.pose_landmarks):
        angr=slope_(results.pose_landmarks.landmark, 16, 14, 12)
        print("right angle: ", angr)
        angl=slope_(results.pose_landmarks.landmark, 11, 13, 15)
        print("left angle: ", angl)



    if (results.face_landmarks and results.right_hand_landmarks):
        # print(results.right_hand_landmarks.landmark[12].z, results.face_landmarks.landmark[130].z)
        if l2_dist(results.face_landmarks.landmark[130], results.right_hand_landmarks.landmark[12], h, w, 1)>1.75*l2_dist(results.right_hand_landmarks.landmark[12], results.right_hand_landmarks.landmark[11], h, w, 1):
            print("Not correct distance.")
        else:
            print("Correct distance.")
        
    mp_drawing.draw_landmarks(
        image,
        results.face_landmarks,
        mp_holistic.FACEMESH_CONTOURS,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp_drawing_styles
        .get_default_face_mesh_contours_style())
    mp_drawing.draw_landmarks(
        image,
        results.pose_landmarks,
        mp_holistic.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles
        .get_default_pose_landmarks_style())
    mp_drawing.draw_landmarks(
        image,
        results.right_hand_landmarks,
        mp_holistic.HAND_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles
        .get_default_pose_landmarks_style())
    # Flip the image horizontally for a selfie-view display.
    cv2.imshow('MediaPipe Holistic', cv2.flip(image, 1))
    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()