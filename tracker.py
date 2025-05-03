import cv2
import numpy as np
import os
import sys

class GolfBallDetector:
    def __init__(self, video_path, template_path, display_mode="original"):
        """Initialize the detector with video and template paths."""
        self.display_mode = "original"  # Options: "original", "grayscale", "edges", "roi"
        self.video_path = "golf_shot.mp4"
        self.template_path = template_path
        self.window_name = f"Golf Ball Detector - {self.display_mode}"
        
        self.contact_point = (0,0)
        self.contact_detected = False
        self.prev_distance = None
        self.max_distance = 0  # Track maximum distance for backswing
        self.prediction_started = False
        self.state = "IDLE"  # States: IDLE, BACKSWING, FORWARD_SWING, TRACKING_BALL
        self.ball_position_history = []  # Store ball positions for flight path
        print("Starting improved shaft and ball detection with backswing and contact detection...")

        # Validate file paths
        if not os.path.exists(self.video_path) or not os.path.exists(template_path):
            print("Error: Video or template file not found.")
            sys.exit(1)
        
        # Load video and template
        self.cap = cv2.VideoCapture(self.video_path)
        self.template = cv2.imread(template_path, 0)
        self.template = cv2.resize(self.template, (int(self.template.shape[1] * 1.5), int(self.template.shape[0] * 1.5)))
        self.t_h, self.t_w = self.template.shape
        
        # Video properties
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        print(f"Video: {self.width}x{self.height} at {self.fps}fps")
        
        # Window setup
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, self.width // 2, self.height // 2)
        
        # Initialize state variables
        self.frame_count = 0
        self.prev_shaft_position = None
        self.ball_position = None
        self.ball_detected = False
        self.initial_ball_position = None
        self.history = []  # Ball detection history
        self.shaft_history = []  # Last 3 shaft lines for smoothing
        self.avg_history = []  # Last 2 averaged shaft positions for prediction
        self.epsilon = 3  # Adjusted for faster detection
        
        print("Starting improved shaft and ball detection with backswing and contact detection...")

    def process_frame(self):
        """Process a single frame from the video."""
        ret, frame = self.cap.read()
        if not ret:
            print(f"Stopped at frame {self.frame_count} (end of video)")
            return False, None

        # Convert to grayscale and HSV
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Filter out grass
        filtered_frame = self.filter_grass(hsv_frame, gray_frame)

        # Edge detection
        blurred = cv2.GaussianBlur(filtered_frame, (5,5), 5)
        edges = cv2.Canny(blurred, 50, 200)

        # Define ROI (using full frame as per your adjustment)
        roi = edges

        # Detect shaft
        current_bottom_shaft, display_frame = self.detect_shaft(roi, frame, gray_frame)

        # Detect ball
        self.detect_ball(gray_frame, current_bottom_shaft)

        # Handle states
        key = None
        if self.state in ["IDLE", "BACKSWING", "FORWARD_SWING"] and current_bottom_shaft and self.ball_position and not self.contact_detected:
            distance = np.sqrt((current_bottom_shaft[0] - self.ball_position[0])**2 + 
                            (current_bottom_shaft[1] - self.ball_position[1])**2)
            contact_threshold = 0  # Actual contact threshold
            pre_contact_threshold = 100  # Slightly larger to catch frame before contact
            
            # Track previous distance for velocity check
            distance_decreasing = False
            if self.prev_distance is not None:
                distance_decreasing = distance < self.prev_distance
            
            print(f"Frame {self.frame_count}: State: {self.state}, Shaft-Ball Distance: {distance:.1f}")

            # State transitions
            if self.state == "IDLE":
                if distance > 500:  # Significant distance indicates backswing start
                    self.state = "BACKSWING"
                    self.max_distance = distance
                    print(f"Frame {self.frame_count}: Transition to BACKSWING")
            
            elif self.state == "BACKSWING":
                self.max_distance = max(self.max_distance, distance)
                if distance_decreasing and self.max_distance > 100:  # Backswing complete, moving toward ball
                    self.state = "FORWARD_SWING"
                    print(f"Frame {self.frame_count}: Transition to FORWARD_SWING, max distance: {self.max_distance:.1f}")

            elif self.state == "FORWARD_SWING":
                # Detect frame just before contact
                if distance <= pre_contact_threshold and distance > contact_threshold and distance_decreasing:
                    print(f"Frame {self.frame_count}: Pre-contact detected! Distance: {distance:.1f}")
                    self.contact_detected = True
                    self.state = "TRACKING_BALL"
                    self.ball_position_history = [self.ball_position]  # Start tracking with current position
                    self.contact_point = self.ball_position
            
            # Store current distance for next frame
            self.prev_distance = distance

            # Display frame and wait for smooth playback
            cv2.imshow(self.window_name, display_frame)
            key = cv2.waitKey(int(1000 / self.fps)) & 0xFF

        elif self.state == "TRACKING_BALL":
            # Track the ball's flight path and get key press
            self.track_ball_flight(gray_frame, display_frame)

            # Display frame and wait for smooth playback
            cv2.imshow(self.window_name, display_frame)
            key = cv2.waitKey(int(1000 / self.fps)) & 0xFF



        # Display frame
        self.display_frame(display_frame)

        self.frame_count += 1
        if self.frame_count % 50 == 0:
            print(f"Processed {self.frame_count} frames...")
        
        return True, key

    def track_ball_flight(self, gray_frame, display_frame):
        """Track the golf ball's flight path in the region above contact point."""
        last_pos = self.ball_position_history[-1]

        if self.frame_count == 145:
            tempvalue = 0

        # Crop the frame to only include the region above contact point
        crop_y_end = max(1, int(self.contact_point[1]))  # Ensure at least 1 pixel height
        if crop_y_end >= self.height or crop_y_end <= self.t_h:
            print(f"TRACK: Frame {self.frame_count}: Invalid crop region (y_end={crop_y_end}), using full frame")
            search_frame = gray_frame
            y_offset = 0
        else:
            search_frame = gray_frame[0:crop_y_end, :]
            y_offset = 0  # Top of the frame, no offset needed

        # Search for ball in the cropped frame
        new_pos = None
        if search_frame.shape[0] >= self.t_h and search_frame.shape[1] >= self.t_w and not self.prediction_started:
            result = cv2.matchTemplate(search_frame, self.template, cv2.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
            if max_val > 0.8:  # Confidence threshold
                new_pos = (max_loc[0] + self.t_w // 2, max_loc[1] + self.t_h // 2 + y_offset)
                print(f"TRACK: Frame {self.frame_count}: Ball tracked at {new_pos}, confidence: {max_val:.2f}")
        else:
            print(f"TRACK: Frame {self.frame_count}: Search frame too small for template matching")

        if new_pos is None:
            print(f"TRACK: Frame {self.frame_count}: Ball not found in cropped region")
            # Use predicted position if available
            if len(self.ball_position_history) > 1:
                prev_pos = self.ball_position_history[-2]
                curr_pos = self.ball_position_history[-1]
                # Apply damping factor to velocity to reduce overshooting
                damping_factor = 0.8
                velocity = (damping_factor * (curr_pos[0] - prev_pos[0]), damping_factor * (curr_pos[1] - prev_pos[1]))
                predicted_pos = (curr_pos[0] + velocity[0], curr_pos[1] + velocity[1])
                new_pos = predicted_pos
                self.prediction_started = True
                print(f"TRACK: Frame {self.frame_count}: Using damped predicted position {new_pos}, damping={damping_factor}")
            else:
                new_pos = last_pos  # Fallback to last position
                self.prediction_started = True
                print(f"TRACK: Frame {self.frame_count}: Using last position {new_pos}")

        # Update ball position and history
        self.ball_position = new_pos
        self.ball_position_history.append(new_pos)

        # Draw red circle around ball (same style as display_frame)
        ball_pos_int = (int(self.ball_position[0]), int(self.ball_position[1]))
        cv2.circle(display_frame, ball_pos_int, self.t_w * 2, (0, 0, 255), 2)

        # Apply Hough Circles on the red circle area only after prediction has started
        if self.prediction_started:
            # Define ROI around ball_pos_int (red circle center)
            roi_size = int(self.t_w * 4)  # ROI size based on red circle radius (self.t_w * 2)
            roi_x = max(0, ball_pos_int[0] - roi_size // 2)
            roi_y = max(0, ball_pos_int[1] - roi_size // 2)
            roi_x2 = min(search_frame.shape[1], roi_x + roi_size)
            roi_y2 = min(search_frame.shape[0], roi_y + roi_size)
            roi = search_frame[roi_y:roi_y2, roi_x:roi_x2]

            if roi.size > 0:
                # Apply Gaussian blur to the ROI
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
                    # Select the best circle (closest to ball_pos_int)
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
                        print(f"TRACK: Frame {self.frame_count}: Hough circle detected at {hough_pos}, radius={best_circle[2]}")
                        # Draw yellow circle
                        cv2.circle(display_frame, (int(hough_pos[0]), int(hough_pos[1])), best_circle[2] * 2, (0, 255, 255), 2)
                else:
                    print(f"TRACK: Frame {self.frame_count}: No Hough circles detected in ROI")

        # Draw the flight path (green)
        for i in range(1, len(self.ball_position_history)):
            pt1 = (int(self.ball_position_history[i-1][0]), int(self.ball_position_history[i-1][1]))
            pt2 = (int(self.ball_position_history[i][0]), int(self.ball_position_history[i][1]))
            cv2.line(display_frame, pt1, pt2, (0, 255, 0), 2)



    def detect_shaft(self, roi, frame, gray_frame):
        """Detect the golf shaft in the roi and return its position."""
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

        if self.display_mode == "original":
            display_frame = frame.copy()
        elif self.display_mode == "grayscale":
            display_frame = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2BGR)
        elif self.display_mode == "edges":
            display_frame = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)
        elif self.display_mode == "roi":
            display_frame = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)

        current_bottom_shaft = None
        if longest_line:
            x1, y1, x2, y2 = longest_line
            current_bottom_shaft = (x2, y2) if y2 > y1 else (x1, y1)
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            print(f"Frame {self.frame_count}: Shaft at {shaft_position}, length={max_length:.1f}, angle={angle:.1f}°, score={best_score:.1f}")
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
            cv2.line(display_frame, (display_x1, display_y1), (display_x2, display_y2), (0, 255, 255), 2)
            display_bottom_shaft = (display_x2, display_y2) if display_y2 > display_y1 else (display_x1, display_y1)
            cv2.circle(display_frame, display_bottom_shaft, 50, (255, 0, 0), 2)
            self.prev_shaft_position = shaft_position

        return current_bottom_shaft, display_frame

    def detect_ball(self, gray_frame, current_bottom_shaft):
        """Detect the golf ball near the shaft position."""
        if current_bottom_shaft and not self.ball_detected:
            roi_size = 200
            roi_x = max(0, current_bottom_shaft[0] - roi_size // 2)
            roi_y = max(0, current_bottom_shaft[1] - roi_size // 2)
            roi_x2 = min(self.width, roi_x + roi_size)
            roi_y2 = min(self.height, roi_y + roi_size)
            ball_roi = gray_frame[roi_y:roi_y2, roi_x:roi_x2]

            if ball_roi.size == 0:
                return

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
                print(f"Frame {self.frame_count}: Ball detected at {current_ball_position}, distance to shaft: {dist:.1f}, confidence: {confidence:.2f}")

                if self.history and len(self.history) >= 1:
                    if dist > self.history[-1][1] + self.epsilon:
                        self.ball_detected = True
                        self.initial_ball_position = self.history[-1][0]
                        print(f"Frame {self.frame_count}: Distance increasing, locking ball at {self.initial_ball_position}")
                    else:
                        self.ball_position = current_ball_position
                        self.history.append((self.ball_position, dist))
                        if len(self.history) > 3:
                            self.history.pop(0)
                        print(f"Frame {self.frame_count}: Distance stable or decreasing, updating ball position")
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
                    print(f"Frame {self.frame_count}: Ball no longer detected, resetting")
                else:
                    self.ball_position = self.initial_ball_position
                    print(f"Frame {self.frame_count}: Ball verified at {self.ball_position}, confidence: {max_val:.2f}")

    def filter_grass(self, hsv_frame, gray_frame):
        """Filter out grass while preserving sunset colors and reducing sky noise."""
        lower_green = np.array([25, 20, 20])
        upper_green = np.array([120, 255, 255])
        grass_mask = cv2.inRange(hsv_frame, lower_green, upper_green)
        sky_height = int(self.height * 0.3)
        sky_region = hsv_frame[:sky_height, :]
        gray_sky = gray_frame[:sky_height, :]
        sobelx = cv2.Sobel(gray_sky, cv2.CV_64F, 1, 0, ksize=5)
        sobely = cv2.Sobel(gray_sky, cv2.CV_64F, 0, 1, ksize=5)
        gradient_mag = np.sqrt(sobelx**2 + sobely**2)
        sky_mask = cv2.threshold(gradient_mag, 20, 255, cv2.THRESH_BINARY_INV)[1].astype(np.uint8)
        full_sky_mask = np.zeros_like(gray_frame, dtype=np.uint8)
        full_sky_mask[:sky_height, :] = sky_mask
        grass_mask = cv2.bitwise_and(grass_mask, cv2.bitwise_not(full_sky_mask))
        blurred_gray = cv2.GaussianBlur(gray_frame, (5, 5), 0)
        variance = cv2.absdiff(gray_frame, blurred_gray)
        texture_mask = cv2.threshold(variance, 15, 255, cv2.THRESH_BINARY)[1].astype(np.uint8)
        grass_mask = cv2.bitwise_and(grass_mask, texture_mask)
        horizon_start = int(self.height * 0.4)
        horizon_end = int(self.height * 0.6)
        horizon_mask = np.ones_like(gray_frame, dtype=np.uint8) * 255
        horizon_mask[horizon_start:horizon_end, :] = 0
        grass_mask = cv2.bitwise_and(grass_mask, horizon_mask)
        kernel = np.ones((3, 3), np.uint8)
        grass_mask = cv2.morphologyEx(grass_mask, cv2.MORPH_OPEN, kernel)
        grass_mask = cv2.morphologyEx(grass_mask, cv2.MORPH_CLOSE, kernel)
        non_grass_mask = cv2.bitwise_not(grass_mask)
        filtered_frame = cv2.bitwise_and(gray_frame, gray_frame, mask=non_grass_mask)
        return filtered_frame

    def display_frame(self, display_frame):
        """Display the frame with overlays."""
        # Add debug text to confirm color frame
        cv2.putText(display_frame, f"State: {self.state}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        if self.ball_position and self.state != "TRACKING_BALL":
            ball_pos_int = (int(self.ball_position[0]), int(self.ball_position[1]))
            cv2.circle(display_frame, ball_pos_int, self.t_w * 2, (0, 0, 255), 2)

        cv2.imshow(self.window_name, display_frame)

    def cleanup(self):
        """Release resources and close windows."""
        self.cap.release()
        cv2.destroyAllWindows()
        print("Program ended.")

    def run(self):
        """Main loop to process video frames."""
        while self.cap.isOpened():
            continue_loop, key = self.process_frame()
            if not continue_loop:
                break
            if key == ord('q'):
                self.cleanup()
                sys.exit(0)
        self.cleanup()

if __name__ == "__main__":
    detector = GolfBallDetector("golf_shot.mp4", "golf_ball_template.png", display_mode="original")
    detector.run()