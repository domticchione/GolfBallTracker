import cv2
import numpy as np

class GrassFilter:
    def __init__(self, width, height):
        """Initialize grass filter with frame dimensions."""
        self.width = width
        self.height = height

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