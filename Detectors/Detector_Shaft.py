import cv2
import numpy as np
from Misc.States import State

class ShaftDetector:
    def __init__(self, width, height):
        """Initialize shaft detector with frame dimensions."""
        self.width = width
        self.height = height
        self.prev_shaft_position = None
        self.shaft_history = []  # Last 3 shaft lines for smoothing
        self.avg_history = []  # Last 2 averaged shaft positions for prediction

    def detect_shaft(self, roi, frame, gray_frame, state, display_mode="original"):
        """Detect the golf shaft in the ROI and draw on the appropriate frame."""
        lines = cv2.HoughLinesP(roi, 1, np.pi / 180, threshold=100, minLineLength=100, maxLineGap=8)
        shaft_position = None
        longest_line = None
        max_length = 0
        best_score = float('inf')

        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2) * 2
                mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
                candidate_pos = (int(mid_x), int(mid_y))
                score = -length
                if self.prev_shaft_position:
                    dist_prev = np.sqrt((candidate_pos[0] - self.prev_shaft_position[0])**2 + 
                                        (candidate_pos[1] - self.prev_shaft_position[1])**2)
                    score += 0.5 * dist_prev
                if score < best_score:
                    best_score = score
                    max_length = length
                    longest_line = (x1, y1, x2, y2)
                    shaft_position = candidate_pos

        # Initialize display frame based on mode
        if display_mode == "original":
            display_frame = frame.copy()
        elif display_mode == "grayscale":
            display_frame = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2BGR)
        elif display_mode == "edges":
            display_frame = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)
        elif display_mode == "roi":
            display_frame = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)
        else:
            display_frame = frame.copy()

        current_bottom_shaft = None
        if longest_line:
            x1, y1, x2, y2 = longest_line
            current_bottom_shaft = (x2, y2) if y2 > y1 else (x1, y1)
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            print(f"ShaftDetector: Shaft at {shaft_position}, length={max_length:.1f}, angle={angle:.1f}°, score={best_score:.1f}")
            self.shaft_history.append((x1, y1, x2, y2))
            if len(self.shaft_history) > 3:
                self.shaft_history.pop(0)
            avg_x1 = sum([line[0] for line in self.shaft_history]) / len(self.shaft_history)
            avg_y1 = sum([line[1] for line in self.shaft_history]) / len(self.shaft_history)
            avg_x2 = sum([line[2] for line in self.shaft_history]) / len(self.shaft_history)
            avg_y2 = sum([line[3] for line in self.shaft_history]) / len(self.shaft_history)
            x1_avg, y1_avg, x2_avg, y2_avg = int(avg_x1), int(avg_y1), int(avg_x2), int(avg_y2)
            self.avg_history.append((x1_avg, y1_avg, x2_avg, y2_avg))
            if len(self.avg_history) > 2:
                self.avg_history.pop(0)
            if len(self.avg_history) == 2:
                prev_x1, prev_y1, prev_x2, prev_y2 = self.avg_history[0]
                curr_x1, curr_y1, curr_x2, curr_y2 = self.avg_history[1]
                vel_x1, vel_y1 = curr_x1 - prev_x1, curr_y1 - prev_y1
                vel_x2, vel_y2 = curr_x2 - prev_x2, curr_y2 - prev_y2
                display_x1, display_y1 = int(curr_x1 + vel_x1), int(curr_y1 + vel_y1)
                display_x2, display_y2 = int(curr_x2 + vel_x2), int(curr_y2 + vel_y2)
            else:
                display_x1, display_y1 = x1_avg, y1_avg
                display_x2, display_y2 = x2_avg, y2_avg

            if state != State.TRACKING_BALL:
                # Draw yellow line for shaft
                cv2.line(display_frame, (display_x1, display_y1), (display_x2, display_y2), (0, 255, 255), 2)
            
            # Draw blue circle at bottom shaft
            display_bottom_shaft = (display_x2, display_y2) if display_y2 > display_y1 else (display_x1, display_y1)
            #cv2.circle(display_frame, display_bottom_shaft, 50, (255, 0, 0), 2)
            self.prev_shaft_position = shaft_position

        return current_bottom_shaft, display_frame