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
            if max_val > 0.8:
                new_pos = (max_loc[0] + self.t_w // 2, max_loc[1] + self.t_h // 2 + y_offset)
                print(f"BallTracker: Frame {frame_count}: Ball tracked at {new_pos}, confidence: {max_val:.2f}")
        else:
            print(f"BallTracker: Frame {frame_count}: Search frame too small for template matching")

        if new_pos is None:
            print(f"BallTracker: Frame {frame_count}: Ball not found in cropped region")
            if len(self.ball_position_history) > 1:
                prev_pos = self.ball_position_history[-2]
                curr_pos = self.ball_position_history[-1]
                damping_factor = 0.8
                velocity = (damping_factor * (curr_pos[0] - prev_pos[0]), damping_factor * (curr_pos[1] - prev_pos[1]))
                predicted_pos = (curr_pos[0] + velocity[0], curr_pos[1] + velocity[1])
                new_pos = predicted_pos
                self.prediction_started = True
                print(f"BallTracker: Frame {frame_count}: Using damped predicted position {new_pos}, damping={damping_factor}")
            else:
                new_pos = last_pos
                self.prediction_started = True
                print(f"BallTracker: Frame {frame_count}: Using last position {new_pos}")

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
            roi_y = max(0, ball_pos_int[1] - roi_size // 2)
            roi_x2 = min(search_frame.shape[1], roi_x + roi_size)
            roi_y2 = min(search_frame.shape[0], roi_y + roi_size)
            roi = search_frame[roi_y:roi_y2, roi_x:roi_x2]

            if roi.size > 0:
                blurred_roi = cv2.GaussianBlur(roi, (5, 5), 0)
                circles = cv2.HoughCircles(
                    blurred_roi,
                    cv2.HOUGH_GRADIENT,
                    dp=10,
                    minDist=20,
                    param1=150,
                    param2=1,
                    minRadius=int(0),
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