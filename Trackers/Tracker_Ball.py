import cv2
import numpy as np
import math

class BallTracker:
    def __init__(self, width, height, template, contact_point, frame_rate):
        """Initialize ball tracker with frame dimensions, template, and contact point."""
        # Store the frame dimensions (width and height of the video frame)
        self.width = width
        self.height = height
        # Store the template image of the ball for template matching
        self.template = template
        # Get the height and width of the template image
        self.t_h, self.t_w = template.shape
        # Store the initial contact point (where the ball starts, e.g., at impact)
        self.contact_point = contact_point
        # Initialize a list to store the history of ball positions, starting with the contact point
        self.ball_position_history = [contact_point]
        # Flag to track whether the tracker has switched to prediction mode
        self.prediction_started = False
        # Initialize the current ball position as the contact point
        self.ball_position = contact_point
        # Initialize the current frame rate of the video
        self.frame_rate = frame_rate

    def track_ball_flight(self, gray_frame, display_frame, frame_count):
        """Track the golf ball's flight path and draw on the provided display frame."""
        # Get the last known position of the ball from the position history
        last_pos = self.ball_position_history[-1]
        
        # Crop the frame to focus only on the region above the contact point (where the ball is expected to move)
        crop_y_end = max(1, int(self.contact_point[1]))
        if crop_y_end >= self.height or crop_y_end <= self.t_h:
            # If the crop region is invalid (e.g., too small or outside frame bounds), use the full frame
            print(f"BallTracker: Frame {frame_count}: Invalid crop region (y_end={crop_y_end}), using full frame")
            search_frame = gray_frame
            y_offset = 0
        else:
            # Crop the frame to the region above the contact point
            search_frame = gray_frame[0:crop_y_end, :]
            y_offset = 0

        # Initialize variable to store the new position of the ball
        new_pos = None
        
        # Perform template matching if the search frame is large enough and prediction mode hasn't started
        if search_frame.shape[0] >= self.t_h and search_frame.shape[1] >= self.t_w and not self.prediction_started:
            # Use normalized cross-correlation (TM_CCOEFF_NORMED) for template matching
            result = cv2.matchTemplate(search_frame, self.template, cv2.TM_CCOEFF_NORMED)
            # Find the minimum and maximum values and their locations in the result
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
            # If the match confidence is above the threshold (0.8), consider it a valid detection
            if max_val > 0.8:
                # Calculate the center of the matched region as the new ball position
                new_pos = (max_loc[0] + self.t_w // 2, max_loc[1] + self.t_h // 2 + y_offset)
                print(f"BallTracker: Frame {frame_count}: Ball tracked at {new_pos}, confidence: {max_val:.2f}")
        else:
            # Log if the search frame is too small for template matching
            print(f"BallTracker: Frame {frame_count}: Search frame too small for template matching")

        # If no new position was found via template matching, predict the position
        if new_pos is None:
            print(f"BallTracker: Frame {frame_count}: Ball not found in cropped region")
            if len(self.ball_position_history) > 1:
                damping_factor = 0.8
                # Use the two most recent positions to estimate velocity
                prev_pos = self.ball_position_history[-2]
                curr_pos = self.ball_position_history[-1]

                scaleFactor = 0.001

                xPix = curr_pos[0] - prev_pos[0]
                yPix = curr_pos[1] - prev_pos[1]

                xMeters = xPix * scaleFactor
                yMeters = yPix * scaleFactor

                displacement = math.sqrt(xMeters**2 + yMeters**2)

                deltaT = 1/self.frame_rate

                velocityMPS = displacement / deltaT

                velocityPix = (xPix * damping_factor, yPix * damping_factor)

                if velocityMPS < 0.1:
                    # Horizontal: Use last known velocity (pixels/frame)
                    vx_pix = velocityPix[0]
                    x_predicted = curr_pos[0] + vx_pix

                    #Vertical: Apply gravity (convert m/s^2 to pixels/frame^2)
                    gravity_pixels = 0.5 * 9.81 * (deltaT ** 2) / scaleFactor
                    y_predicted = curr_pos[1] + velocityPix[1] + gravity_pixels

                    predicted_pos = (x_predicted, y_predicted)

                else:
                    # Predict the next position based on current position and damped velocity
                    predicted_pos = (curr_pos[0] + velocityPix[0], curr_pos[1] + velocityPix[1])

                new_pos = predicted_pos
                # Switch to prediction mode once template matching fails
                self.prediction_started = True
                print(f"BallTracker: Frame {frame_count}: Using damped predicted position {new_pos}, damping={damping_factor}")
            else:
                # If there's only one position in history, use the last known position
                new_pos = last_pos
                self.prediction_started = True
                print(f"BallTracker: Frame {frame_count}: Using last position {new_pos}")

        # Draw a red circle around the previous ball position on the display frame
        ball_pos_int = (int(new_pos[0]), int(new_pos[1]))
        cv2.circle(display_frame, ball_pos_int, self.t_w * 2, (0, 0, 255), 2)

        # Apply Hough Circle detection in a region of interest (ROI) around the predicted position
        if self.prediction_started:
            # Define the size of the ROI (4 times the template width)
            roi_size = int(self.t_w * 2)
            # Calculate ROI boundaries, ensuring they stay within the frame
            roi_x = max(0, ball_pos_int[0] - roi_size // 2)
            roi_y = max(0, ball_pos_int[1] - roi_size // 2)
            roi_x2 = min(search_frame.shape[1], roi_x + roi_size)
            roi_y2 = min(search_frame.shape[0], roi_y + roi_size)
            # Extract the ROI from the search frame
            roi = search_frame[roi_y:roi_y2, roi_x:roi_x2]

            # Perform Hough Circle detection if the ROI is valid
            if roi.size > 0:
                # Apply Gaussian blur to reduce noise in the ROI
                blurred_roi = cv2.GaussianBlur(roi, (5, 5), 0)
                # Detect circles using HoughCircles with specified parameters
                circles = cv2.HoughCircles(
                    blurred_roi,
                    cv2.HOUGH_GRADIENT,
                    dp=10,  # Inverse ratio of resolution
                    minDist=20,  # Minimum distance between detected circles
                    param1=150,  # Canny edge detector threshold
                    param2=1,  # Accumulator threshold for circle detection
                    minRadius=0,  # Minimum circle radius
                    maxRadius=int(self.t_w * 0.2)  # Maximum circle radius
                )
                if circles is not None:
                    # Round circle coordinates and convert to integers
                    circles = np.round(circles[0, :]).astype("int")
                    best_circle = None
                    min_dist = float(30)
                    # Find the circle closest to the predicted position
                    for (x, y, r) in circles:
                        circle_pos = (x + roi_x, y + roi_y)
                        dist = np.sqrt((circle_pos[0] - ball_pos_int[0])**2 + (circle_pos[1] - ball_pos_int[1])**2)
                        if dist < min_dist:
                            min_dist = dist
                            #best_circle = (circle_pos[0], circle_pos[1], r)
                    if best_circle:
                        # If a valid circle is found, draw a yellow circle around it
                        hough_pos = (best_circle[0], best_circle[1])
                        new_pos = hough_pos
                        
                        print(f"BallTracker: Frame {frame_count}: Hough circle detected at {hough_pos}, radius={best_circle[2]}")
                        cv2.circle(display_frame, (int(hough_pos[0]), int(hough_pos[1])), best_circle[2] * 2, (0, 255, 255), 2)
                else:
                    print(f"BallTracker: Frame {frame_count}: No Hough circles detected in ROI")

        # Update position history with new_pos (either template match, prediction, or Hough)
        self.ball_position = new_pos
        self.ball_position_history.append(new_pos)

        # Draw smooth trajectory using position history
        for i in range(1, len(self.ball_position_history)):
            pt1 = (int(self.ball_position_history[i-1][0]), int(self.ball_position_history[i-1][1]))
            pt2 = (int(self.ball_position_history[i][0]), int(self.ball_position_history[i][1]))
            cv2.line(display_frame, pt1, pt2, (0, 255, 0), 2)

        # Return the current ball position and the modified display frame
        return self.ball_position, display_frame