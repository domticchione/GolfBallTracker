import cv2
import numpy as np
from States import State

class Detector_Ball:
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
        self.history_for_tracker = []
        self.detection_count = 0  # Track consecutive detections for locking

    def detect_ball(self, gray_frame, current_bottom_shaft, display_frame, state):
        """Detect the golf ball using template matching followed by Hough circles, with stronger weighting towards club head."""
        if current_bottom_shaft and not self.ball_detected and state != State.TRACKING_BALL:
            # Use the full frame for template matching
            search_frame = gray_frame

            # Perform template matching
            new_pos = None
            if search_frame.shape[0] >= self.t_h and search_frame.shape[1] >= self.t_w:
                result = cv2.matchTemplate(search_frame, self.template, cv2.TM_CCOEFF_NORMED)
                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
                if max_val > 0.8:
                    new_pos = (max_loc[0] + self.t_w // 2, max_loc[1] + self.t_h // 2)
                    print(f"BallDetector: Template match at {new_pos}, confidence: {max_val:.2f}")
                else:
                    print(f"BallDetector: Template match confidence too low: {max_val:.2f}")
            else:
                print("BallDetector: Frame too small for template matching")

            # If template match found, apply Hough circles in ROI
            if new_pos:
                # Define larger ROI around template match position
                roi_size = int(self.t_w * 6)
                roi_x = max(0, int(new_pos[0]) - roi_size // 2)
                roi_y = max(0, int(new_pos[1]) - roi_size // 2)
                roi_x2 = min(self.width, roi_x + roi_size)
                roi_y2 = min(self.height, roi_y + roi_size)
                roi = gray_frame[roi_y:roi_y2, roi_x:roi_x2]

                if roi.size > 0:
                    blurred_roi = cv2.GaussianBlur(roi, (5, 5), 0)
                    circles = cv2.HoughCircles(
                        blurred_roi,
                        cv2.HOUGH_GRADIENT,
                        dp=2,
                        minDist=20,
                        param1=100,
                        param2=1,
                        minRadius=int(self.t_w * 0.3),
                        maxRadius=int(self.t_w * 0.7)
                    )
                    ball_candidates = []
                    if circles is not None:
                        circles = np.round(circles[0, :]).astype("int")
                        # Draw all circles for debugging (blue)
                        for (x, y, r) in circles:
                            cv2.circle(display_frame, (x + roi_x, y + roi_y), r, (255, 0, 0), 1)
                        for (x, y, r) in circles:
                            circle_pos = (x + roi_x, y + roi_y)
                            # Compute distance to club head
                            dist_shaft = np.sqrt((circle_pos[0] - current_bottom_shaft[0])**2 + 
                                                 (circle_pos[1] - current_bottom_shaft[1])**2)
                            # Compute distance to template match center
                            dist_template = np.sqrt((circle_pos[0] - new_pos[0])**2 + 
                                                    (circle_pos[1] - new_pos[1])**2)
                            # Determine relative position to clubhead
                            rel_x = circle_pos[0] - current_bottom_shaft[0]
                            rel_y = circle_pos[1] - current_bottom_shaft[1]
                            rel_pos = ("right" if rel_x > 0 else "left") + ", " + ("below" if rel_y > 0 else "above")
                            # Score prioritizes radius, with stronger shaft distance and template proximity penalties
                            score = -r + 0.05 * dist_shaft + 0.02 * dist_template
                            ball_candidates.append((circle_pos, r, score, dist_shaft, rel_pos))
                            print(f"BallDetector: Circle at {circle_pos}, radius={r}, shaft dist: {dist_shaft:.1f}, template dist: {dist_template:.1f}, relative: {rel_pos}, score: {score:.2f}")
                    
                    if ball_candidates:
                        # Sort by score (strongest circle, with distances as tiebreakers)
                        ball_candidates.sort(key=lambda x: x[2])
                        best_circle = ball_candidates[0]
                        new_pos = best_circle[0]
                        radius = best_circle[1]
                        dist_shaft = best_circle[3]
                        rel_pos = best_circle[4]
                        print(f"BallDetector: Selected Hough circle at {new_pos}, radius={radius}, shaft dist: {dist_shaft:.1f}, relative: {rel_pos}")
                    else:
                        print("BallDetector: No Hough circles detected in ROI")
                        new_pos = None
                else:
                    print("BallDetector: Invalid ROI for Hough circles")
                    new_pos = None
            else:
                print("BallDetector: No template match found")
                new_pos = None

            # Update ball position and history
            if new_pos:
                self.ball_position = new_pos
                if self.history and len(self.history) >= 1:
                    prev_pos = self.history[-1][0]
                    dist = np.sqrt((new_pos[0] - prev_pos[0])**2 + (new_pos[1] - prev_pos[1])**2)
                    if dist < 50:  # Consistency threshold (pixels)
                        self.detection_count += 1
                        if self.detection_count >= 3:  # Require 3 consistent detections
                            self.ball_detected = True
                            self.initial_ball_position = prev_pos
                            print(f"BallDetector: Consistent detections, locking ball at {self.initial_ball_position}")
                    else:
                        self.detection_count = 1
                        print("BallDetector: Inconsistent position, resetting detection count")
                    self.history.append((self.ball_position, radius if 'radius' in locals() else self.t_w // 2))
                    if len(self.history) > 3:
                        self.history.pop(0)
                else:
                    self.detection_count = 1
                    self.history.append((self.ball_position, radius if 'radius' in locals() else self.t_w // 2))
            else:
                self.ball_position = None
                self.history = []
                self.detection_count = 0
                print("BallDetector: No ball detected")

        if self.ball_detected and self.initial_ball_position:
            # Verify ball at locked position using Hough circles
            roi_size = 50
            roi_x = max(0, self.initial_ball_position[0] - roi_size // 2)
            roi_y = max(0, self.initial_ball_position[1] - roi_size // 2)
            roi_x2 = min(self.width, roi_x + roi_size)
            roi_y2 = min(self.height, roi_y + roi_size)
            ball_roi = gray_frame[roi_y:roi_y2, roi_x:roi_x2]

            if ball_roi.size > 0:
                # Apply adaptive thresholding for verification
                roi_adaptive = cv2.adaptiveThreshold(
                    ball_roi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
                )
                blurred_roi = cv2.GaussianBlur(roi_adaptive, (5, 5), 0)
                circles = cv2.HoughCircles(
                    blurred_roi,
                    cv2.HOUGH_GRADIENT,
                    dp=10,
                    minDist=20,
                    param1=100,
                    param2=10,
                    minRadius=int(self.t_w * 0.3),
                    maxRadius=int(self.t_w * 0.7)
                )
                if circles is None:
                    self.ball_detected = False
                    self.ball_position = None
                    self.initial_ball_position = None
                    self.history_for_tracker = self.history
                    self.history = []
                    self.detection_count = 0
                    print("BallDetector: No circles detected at locked position, resetting")
                else:
                    # Select the closest circle to the initial position
                    circles = np.round(circles[0, :]).astype("int")
                    best_circle = None
                    min_dist = float('inf')
                    for (x, y, r) in circles:
                        circle_pos = (x + roi_x, y + roi_y)
                        dist = np.sqrt((circle_pos[0] - self.initial_ball_position[0])**2 + 
                                       (circle_pos[1] - self.initial_ball_position[1])**2)
                        if dist < min_dist:
                            min_dist = dist
                            best_circle = (circle_pos[0], circle_pos[1], r)
                    if best_circle:
                        self.ball_position = (best_circle[0], best_circle[1])
                        print(f"BallDetector: Ball verified at {self.ball_position}, radius: {best_circle[2]}")
                    else:
                        self.ball_detected = False
                        self.ball_position = None
                        self.initial_ball_position = None
                        self.history_for_tracker = self.history
                        self.history = []
                        self.detection_count = 0
                        print("BallDetector: No valid circle near locked position, resetting")

        # Draw yellow circle if ball is detected (matching Tracker_Ball.py)
        if self.ball_position and state != State.TRACKING_BALL:
            ball_pos_int = (int(self.ball_position[0]), int(self.ball_position[1]))
            cv2.circle(display_frame, ball_pos_int, int(self.t_w * 0.5), (0, 255, 255), 2)

        return self.ball_position, display_frame