import cv2
import numpy as np
from Misc.States import State

class BallDetector:
    def __init__(self, width, height, template):
        """Initialize ball detector with frame dimensions and template."""
        self.width = width
        self.height = height
        self.template = template
        self.t_h, self.t_w = template.shape
        self.ball_detected = False
        self.initial_ball_position = None
        self.ball_position = None
        self.history = []  # Ball detection history
        self.epsilon = 3  # For distance comparison

    def detect_ball(self, gray_frame, current_bottom_shaft, display_frame, state):
        """Detect the golf ball near the shaft position and draw red circle on provided frame."""
        if current_bottom_shaft and not self.ball_detected:
            roi_size = 200
            roi_x = max(0, current_bottom_shaft[0] - roi_size // 2)
            roi_y = max(0, current_bottom_shaft[1] - roi_size // 2)
            roi_x2 = min(self.width, roi_x + roi_size)
            roi_y2 = min(self.height, roi_y + roi_size)
            ball_roi = gray_frame[roi_y:roi_y2, roi_x:roi_x2]

            if ball_roi.size == 0:
                return None, display_frame

            result = cv2.matchTemplate(ball_roi, self.template, cv2.TM_CCOEFF_NORMED)
            threshold = 0.5
            locations = np.where(result >= threshold)
            ball_candidates = []

            for pt in zip(*locations[::-1]):
                ball_center = (pt[0] + self.t_w // 2 + roi_x, pt[1] + self.t_h // 2 + roi_y)
                dist = np.sqrt((ball_center[0] - current_bottom_shaft[0])**2 + 
                               (ball_center[1] - current_bottom_shaft[1])**2)
                ball_candidates.append((ball_center, dist, result[pt[1], pt[0]]))

            if ball_candidates:
                ball_candidates.sort(key=lambda x: x[1])
                current_ball_position, dist, confidence = ball_candidates[0]
                current_ball_position = (int(current_ball_position[0]), int(current_ball_position[1]))
                print(f"BallDetector: Ball detected at {current_ball_position}, distance to shaft: {dist:.1f}, confidence: {confidence:.2f}")

                if self.history and len(self.history) >= 1:
                    if dist > self.history[-1][1] + self.epsilon:
                        self.ball_detected = True
                        self.initial_ball_position = self.history[-1][0]
                        print(f"BallDetector: Distance increasing, locking ball at {self.initial_ball_position}")
                    else:
                        self.ball_position = current_ball_position
                        self.history.append((self.ball_position, dist))
                        if len(self.history) > 3:
                            self.history.pop(0)
                        print(f"BallDetector: Distance stable or decreasing, updating ball position")
                else:
                    self.ball_position = current_ball_position
                    self.history.append((self.ball_position, dist))
            else:
                self.ball_position = None
                self.history = []

        if self.ball_detected and self.initial_ball_position:
            roi_size = 50
            roi_x = max(0, self.initial_ball_position[0] - roi_size // 2)
            roi_y = max(0, self.initial_ball_position[1] - roi_size // 2)
            roi_x2 = min(self.width, roi_x + roi_size)
            roi_y2 = min(self.height, roi_y + roi_size)
            ball_roi = gray_frame[roi_y:roi_y2, roi_x:roi_x2]

            if ball_roi.size > 0:
                result = cv2.matchTemplate(ball_roi, self.template, cv2.TM_CCOEFF_NORMED)
                max_val = cv2.minMaxLoc(result)[1]
                if max_val < 0.4:
                    self.ball_detected = False
                    self.ball_position = None
                    self.initial_ball_position = None
                    self.history = []
                    print(f"BallDetector: Ball no longer detected, resetting")
                else:
                    self.ball_position = self.initial_ball_position
                    print(f"BallDetector: Ball verified at {self.ball_position}, confidence: {max_val:.2f}")

        # Draw red circle if ball is detected
        if self.ball_position:
            ball_pos_int = (int(self.ball_position[0]), int(self.ball_position[1]))
            if state != State.TRACKING_BALL:
                cv2.circle(display_frame, ball_pos_int, self.t_w, (0, 0, 255), 2)

        return self.ball_position, display_frame