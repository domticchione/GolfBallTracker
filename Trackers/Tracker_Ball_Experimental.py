import cv2
import numpy as np

class BallTracker:
    def __init__(self, width, height, template, contact_point):
        """Initialize ball tracker with frame dimensions, template, and contact point."""
        self.width = width
        self.height = height
        self.template = template
        self.t_h, self.t_w = template.shape
        self.contact_point = contact_point
        self.ball_position_history = [contact_point]  # Start with contact point
        self.prediction_started = False
        self.ball_position = contact_point
        self.initial_velocity = None  # To store initial velocity for prediction
        self.time_step = 0  # Time step for physics-based prediction
        self.gravity = 9.8 * 30  # Gravity in pixels per frame^2 (scaled for video)
        self.ground_y = contact_point[1]  # Assume ground is at contact point's y-coordinate

    def track_ball_flight(self, gray_frame, display_frame, frame_count):
        """Track the golf ball's flight path and draw on the provided display frame."""
        last_pos = self.ball_position_history[-1]

        # Crop the frame to only include the region above contact point
        crop_y_end = max(1, int(self.contact_point[1]))
        if crop_y_end >= self.height or crop_y_end <= self.t_h:
            print(f"BallTracker: Frame {frame_count}: Invalid crop region (y_end={crop_y_end}), using full frame")
            search_frame = gray_frame
            y_offset = 0
        else:
            search_frame = gray_frame[0:crop_y_end, :]
            y_offset = 0

        # Search for ball in the cropped frame
        new_pos = None
        if search_frame.shape[0] >= self.t_h and search_frame.shape[1] >= self.t_w and not self.prediction_started:
            result = cv2.matchTemplate(search_frame, self.template, cv2.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
            if max_val > 0.6:  # Lowered threshold for better detection
                new_pos = (max_loc[0] + self.t_w // 2, max_loc[1] + self.t_h // 2 + y_offset)
                print(f"BallTracker: Frame {frame_count}: Ball tracked at {new_pos}, confidence: {max_val:.2f}")

                # Estimate initial velocity when we have enough points
                if len(self.ball_position_history) >= 2:
                    prev_pos = self.ball_position_history[-2]
                    self.initial_velocity = (
                        new_pos[0] - prev_pos[0],  # x-velocity (pixels per frame)
                        new_pos[1] - prev_pos[1]   # y-velocity (pixels per frame)
                    )
        else:
            print(f"BallTracker: Frame {frame_count}: Search frame too small for template matching")

        if new_pos is None or self.prediction_started:
            print(f"BallTracker: Frame {frame_count}: Ball not found or prediction started, using prediction")
            self.prediction_started = True
            if self.initial_velocity is None and len(self.ball_position_history) > 1:
                prev_pos = self.ball_position_history[-2]
                curr_pos = self.ball_position_history[-1]
                self.initial_velocity = (
                    curr_pos[0] - prev_pos[0],
                    curr_pos[1] - prev_pos[1]
                )

            if self.initial_velocity:
                # Physics-based prediction: x = x0 + vx * t, y = y0 + vy * t + 0.5 * g * t^2
                self.time_step += 1
                t = self.time_step / 30.0  # Assume 30 fps for time scaling
                predicted_x = last_pos[0] + self.initial_velocity[0] * t
                predicted_y = last_pos[1] + self.initial_velocity[1] * t + 0.5 * self.gravity * t * t

                # Stop prediction if the ball reaches or passes the ground
                if predicted_y >= self.ground_y:
                    predicted_y = self.ground_y
                    print(f"BallTracker: Frame {frame_count}: Ball reached ground at y={predicted_y}")
                    new_pos = (predicted_x, predicted_y)
                    self.ball_position_history.append(new_pos)
                    self.ball_position = new_pos
                    # Draw final position and return
                    cv2.circle(display_frame, (int(new_pos[0]), int(new_pos[1])), self.t_w * 2, (0, 0, 255), 2)
                    cv2.putText(display_frame, "Ball Landed", (int(new_pos[0]) + 10, int(new_pos[1]) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    for i in range(1, len(self.ball_position_history)):
                        pt1 = (int(self.ball_position_history[i-1][0]), int(self.ball_position_history[i-1][1]))
                        pt2 = (int(self.ball_position_history[i][0]), int(self.ball_position_history[i][1]))
                        cv2.line(display_frame, pt1, pt2, (0, 255, 0), 2)
                    return new_pos, display_frame
                else:
                    new_pos = (predicted_x, predicted_y)
                    print(f"BallTracker: Frame {frame_count}: Predicted position {new_pos}, time={t:.2f}s")
            else:
                new_pos = last_pos
                print(f"BallTracker: Frame {frame_count}: Using last position {new_pos} (no velocity)")

        # Update ball position and history
        self.ball_position = new_pos
        self.ball_position_history.append(new_pos)

        # Draw red circle around ball
        ball_pos_int = (int(self.ball_position[0]), int(self.ball_position[1]))
        cv2.circle(display_frame, ball_pos_int, self.t_w * 2, (0, 0, 255), 2)

        # Apply Hough Circles on the red circle area only after prediction has started
        if self.prediction_started:
            roi_size = int(self.t_w * 4)
            roi_x = max(0, ball_pos_int[0] - roi_size // 2)
            roi_y = max(0, ball_pos_int[1] - roi_size)
            roi_x2 = min(search_frame.shape[1], roi_x + roi_size)
            roi_y2 = min(search_frame.shape[0], roi_y + roi_size)
            roi = search_frame[roi_y:roi_y2, roi_x:roi_x2]

            if roi.size > 0:
                blurred_roi = cv2.GaussianBlur(roi, (5, 5), 0)
                circles = cv2.HoughCircles(
                    blurred_roi,
                    cv2.HOUGH_GRADIENT,
                    dp=10,  # Adjusted for better detection
                    minDist=20,
                    param1=150,  # Lowered for sensitivity
                    param2=1,   # Lowered for sensitivity
                    minRadius=0,
                    maxRadius=int(self.t_w * 0.5)
                )
                if circles is not None:
                    circles = np.round(circles[0, :]).astype("int")
                    best_circle = None
                    min_dist = float('inf')
                    for (x, y, r) in circles:
                        circle_pos = (x + roi_x, y + roi_y)
                        dist = np.sqrt((circle_pos[0] - ball_pos_int[0])**2 + (circle_pos[1] - ball_pos_int[1])**2)
                        if dist < min_dist:
                            min_dist = dist
                            best_circle = (circle_pos[0], circle_pos[1], r)
                    if best_circle:
                        hough_pos = (best_circle[0], best_circle[1])
                        print(f"BallTracker: Frame {frame_count}: Hough circle detected at {hough_pos}, radius={best_circle[2]}")
                        # Draw yellow circle
                        cv2.circle(display_frame, (int(hough_pos[0]), int(hough_pos[1])), best_circle[2] * 2, (0, 255, 255), 2)
                else:
                    print(f"BallTracker: Frame {frame_count}: No Hough circles detected in ROI")

        # Draw the flight path (green)
        for i in range(1, len(self.ball_position_history)):
            pt1 = (int(self.ball_position_history[i-1][0]), int(self.ball_position_history[i-1][1]))
            pt2 = (int(self.ball_position_history[i][0]), int(self.ball_position_history[i][1]))
            cv2.line(display_frame, pt1, pt2, (0, 255, 0), 2)

        return self.ball_position, display_frame