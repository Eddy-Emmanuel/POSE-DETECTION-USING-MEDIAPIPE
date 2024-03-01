import cv2
import numpy as np
from mediapipe import solutions

pose_var = solutions.pose
pose_detector = pose_var.Pose()
drawing_tool = solutions.drawing_utils

cam = cv2.VideoCapture(0)

loop = True
while loop:
    _, frame = cam.read()
    h, w, _ = frame.shape
    processed_frame = pose_detector.process(frame)
    drawing_tool.draw_landmarks(frame,
                                processed_frame.pose_landmarks,
                                pose_var.POSE_CONNECTIONS, 
                                drawing_tool.DrawingSpec(thickness=2, circle_radius=3, color=(225, 0, 0)),
                                drawing_tool.DrawingSpec(thickness=2, circle_radius=3, color=(0, 255, 0)))

    empty_board = np.zeros((h, w, _))
    empty_board.fill(255)

    drawing_tool.draw_landmarks(empty_board,
                                processed_frame.pose_landmarks,
                                pose_var.POSE_CONNECTIONS, 
                                drawing_tool.DrawingSpec(thickness=2, circle_radius=3, color=(225, 0, 0)),
                                drawing_tool.DrawingSpec(thickness=2, circle_radius=3, color=(0, 255, 0)))

    cv2.imshow("Pose", frame)
    cv2.imshow("Pose", empty_board)

    if cv2.waitKey(1) == ord("q"):
        loop = False

cam.release()
cv2.destroyAllWindows()