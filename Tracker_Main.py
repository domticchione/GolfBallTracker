import cv2
import numpy as np
import os
import sys
from Detectors.Detector_Shaft import ShaftDetector
from Detectors.Detector_Ball import BallDetector
from Trackers.Tracker_Ball import BallTracker
from Misc.Grass_Filter import GrassFilter
from Misc.States import State

class GolfBallDetector:
    def __init__(self, video_path, template_path, display_mode="original"):
        """Initialize the detector with video and template paths."""
        self.display_mode = display_mode  # Options: "original", "grayscale", "edges", "roi"
        self.video_path = ".\Videos\\" + video_path
        self.template_path = ".\Templates\\" + template_path
        self.window_name = f"Golf Ball Detector - {self.display_mode}"
        self.frame_rate = 30
        self.contact_point = (0,0)
        self.contact_detected = False
        self.prev_distance = None
        self.max_distance = 0
        self.state = State.IDLE
        self.distance_decrease_tolerance = 5.0  # Minimum distance decrease (pixels) per frame
        self.distance_history = []  # Store last two distances for consecutive decrease check
        print("Starting golf ball detection...")

        # Validate file paths
        if not os.path.exists(self.video_path) or not os.path.exists(self.template_path):
            print("Error: Video or template file not found.")
            sys.exit(1)
        
        # Load video and template
        self.cap = cv2.VideoCapture(self.video_path)
        self.template = cv2.imread(self.template_path, 0)
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
        
        # Initialize modules
        self.shaft_detector = ShaftDetector(self.width, self.height)
        self.ball_detector = BallDetector(self.width, self.height, self.template)
        self.grass_filter = GrassFilter(self.width, self.height)
        self.ball_tracker = None  # Initialized when contact is detected
        
        # Frame counter
        self.frame_count = 0

    def process_frame(self):
        """Process a single frame from the video."""
        ret, frame = self.cap.read()
        if not ret:
            print(f"Stopped at frame {self.frame_count} (end of video)")
            return False, None

        # Convert to grayscale and HSV for detection
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Filter out grass
        filtered_frame = self.grass_filter.filter_grass(hsv_frame, gray_frame)

        # Edge detection
        blurred = cv2.GaussianBlur(filtered_frame, (5,5), 5)
        edges = cv2.Canny(blurred, 50, 200)

        # Define ROI (full frame)
        roi = edges

        # Detect shaft
        current_bottom_shaft, display_frame = self.shaft_detector.detect_shaft(roi, frame, gray_frame, self.state, self.display_mode)

        # Detect ball, passing the shaft display frame to accumulate visualizations
        ball_position, display_frame = self.ball_detector.detect_ball(gray_frame, current_bottom_shaft, display_frame, self.state)

        # Handle states
        key = None
        if self.state in [State.IDLE, State.BACKSWING, State.FORWARD_SWING] and current_bottom_shaft and ball_position and not self.contact_detected:
            distance = np.sqrt((current_bottom_shaft[0] - ball_position[0])**2 + 
                              (current_bottom_shaft[1] - ball_position[1])**2)
            contact_threshold = 0
            pre_contact_threshold = 100
            
            # Check for two consecutive decreases
            distance_decreasing = False
            if len(self.distance_history) >= 2:
                curr_distance = distance
                prev_distance = self.distance_history[-1]
                prev_prev_distance = self.distance_history[-2]
                if (curr_distance < prev_distance - self.distance_decrease_tolerance and 
                    prev_distance < prev_prev_distance - self.distance_decrease_tolerance):
                    distance_decreasing = True
            
            # Update distance history after the check
            self.distance_history.append(distance)
            if len(self.distance_history) > 2:
                self.distance_history.pop(0)  # Keep only the last two distances
            
            print(f"Frame {self.frame_count}: State: {self.state.name}, "
                  f"Shaft: {current_bottom_shaft}, Ball: {ball_position}, "
                  f"Distance: {distance:.1f}, History: {[f'{d:.1f}' for d in self.distance_history]}, "
                  f"Decreasing: {distance_decreasing}, Tolerance: {self.distance_decrease_tolerance}")

            if self.state == State.IDLE:
                if distance > 500:
                    self.state = State.BACKSWING
                    self.max_distance = distance
                    print(f"Frame {self.frame_count}: Transition to {State.BACKSWING.name}")
            
            elif self.state == State.BACKSWING:
                self.max_distance = max(self.max_distance, distance)
                if distance_decreasing and self.max_distance > 100:
                    self.state = State.FORWARD_SWING
                    print(f"Frame {self.frame_count}: Transition to {State.FORWARD_SWING.name}, max distance: {self.max_distance:.1f}")

            elif self.state == State.FORWARD_SWING:
                if distance <= pre_contact_threshold and distance > contact_threshold and distance_decreasing:
                    print(f"Frame {self.frame_count}: Pre-contact detected! Distance: {distance:.1f}")
                    self.contact_detected = True
                    self.state = State.TRACKING_BALL
                    self.contact_point = ball_position
                    self.ball_tracker = BallTracker(self.width, self.height, self.template, self.contact_point, self.frame_rate)
            
            self.prev_distance = distance

        elif self.state == State.TRACKING_BALL and self.ball_tracker:
            # Track ball flight, passing the current display frame
            tracked_ball_position, display_frame = self.ball_tracker.track_ball_flight(gray_frame, display_frame, self.frame_count)

        # Add debug text
        cv2.putText(display_frame, f"State: {self.state.name}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Display frame
        cv2.imshow(self.window_name, display_frame)
        key = cv2.waitKey(int(1000 / self.fps)) & 0xFF

        self.frame_count += 1
        if self.frame_count % 50 == 0:
            print(f"Processed {self.frame_count} frames...")
        
        return True, key
    
    def get_framerate(self, video_path):
        try:
            # Open the video file
            video = cv2.VideoCapture(video_path)
        
            # Check if video opened successfully
            if not video.isOpened():
                print("Error: Could not open video file")
                return None
            
            # Get the framerate
            fps = video.get(cv2.CAP_PROP_FPS)
        
            # Release the video object
            video.release()
        
            return fps
        except Exception as e:
            print(f"Error: {str(e)}")
            return None

    def cleanup(self):
        """Release resources and close windows."""
        self.cap.release()
        cv2.destroyAllWindows()
        print("Program ended.")

    def run(self):
        """Main loop to process video frames."""
        self.frame_rate = self.get_framerate(self.video_path)
        while self.cap.isOpened():
            continue_loop, key = self.process_frame()
            if not continue_loop:
                break
            if key == ord('q'):
                self.cleanup()
                sys.exit(0)
        self.cleanup()

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python Tracker_Main.py <video_path> <template_path> [display_mode]")
        sys.exit(1)
    
    video_path = sys.argv[1]
    template_path = sys.argv[2]
    display_mode = sys.argv[3] if len(sys.argv) > 3 else "original"
    
    detector = GolfBallDetector(video_path, template_path, display_mode)
    detector.run()