import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk  # PIL for image processing

class BirdVelocityAnalyzer:
    def __init__(self):
        self.cap = None
        self.current_frame = None
        self.previous_frame_gray = None
        self.previous_position = None
        self.current_position = None
        self.bounding_box = None  # Store the bounding box of the bird
        self.root = None
        self.slider = None
        self.video_label = None
        self.info_label = None
        self.scale_factor = 0.001 / 1000

    def load_video(self, filepath):
        self.cap = cv2.VideoCapture(filepath)
        if not self.cap.isOpened():
            print("Error opening video file")
            return False
        return True

    def process_frame(self, frame_no):
        if not self.cap:
            print("No video loaded")
            return False

        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
        ret, frame = self.cap.read()
        if not ret:
            print("Failed to read frame")
            return False

        self.current_frame = frame
        if self.previous_frame_gray is not None:
            position, bbox = self.find_bird_position(frame)
            self.current_position = position
            if self.previous_position and self.current_position:
                velocity_kmph = self.calculate_velocity()  # This now returns velocity in km/h
                velocity_text = f"Velocity: {velocity_kmph:.2f} km/h"
            if bbox is not None:
                # Draw the bounding box
                cv2.rectangle(self.current_frame, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), (0, 255, 0), 2)

        self.display_frame(self.current_frame)  # Update GUI

        self.previous_position = self.current_position
        if frame is not None:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            self.previous_frame_gray = cv2.GaussianBlur(gray, (21, 21), 0)

        self.info_label.config(text=f"Frame: {frame_no}, {velocity_text}")
        return True

    def display_frame(self, frame):
        if frame is None:
            return

        cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(cv2image)
        imgtk = ImageTk.PhotoImage(image=img)
        self.video_label.imgtk = imgtk
        self.video_label.config(image=imgtk)

    def find_bird_position(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        frame_delta = cv2.absdiff(self.previous_frame_gray, gray)
        thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]

        thresh = cv2.dilate(thresh, None, iterations=2)
        contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        self.previous_frame_gray = gray

        if contours:
            max_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(max_contour)
            return ((x + w // 2, y + h // 2), (x, y, w, h))  # Return the center and bounding box

        return (None, None)

    def calculate_velocity(self):
        if self.previous_position and self.current_position:
            pixel_distance = np.sqrt((self.current_position[0] - self.previous_position[0])**2 + 
                                     (self.current_position[1] - self.previous_position[1])**2)
            time_seconds = 1 / self.cap.get(cv2.CAP_PROP_FPS)
            velocity_kmps = pixel_distance * self.scale_factor / time_seconds
            velocity_kmph = velocity_kmps * 3600
            return velocity_kmph
        return 0

    def on_slider_update(self, val):
        frame_no = int(val)
        self.process_frame(frame_no)

    def create_gui(self):
        self.root = tk.Tk()
        self.root.title("Bird Velocity Analyzer")

        self.video_label = tk.Label(self.root)
        self.video_label.pack()

        self.info_label = tk.Label(self.root, text="Velocity: N/A, Frame: N/A")
        self.info_label.pack()

        load_button = tk.Button(self.root, text="Load Video", command=self.load_video_dialog)
        load_button.pack()

        self.slider = tk.Scale(self.root, from_=0, to=100, orient=tk.HORIZONTAL, command=self.on_slider_update)
        self.slider.pack(fill=tk.X, expand=True)

        self.root.mainloop()

    def load_video_dialog(self):
        file_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4 *.avi *.mov")])
        if not file_path:
            print("No file selected.")
            return

        if self.load_video(file_path):
            video_length = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.slider.config(to=video_length-1)
            ret, frame = self.cap.read()
            if ret:
                self.display_frame(frame)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                self.previous_frame_gray = cv2.GaussianBlur(gray, (21, 21), 0)
                print("Video loaded successfully.")
            else:
                print("Failed to read the first frame.")

# Usage
analyzer = BirdVelocityAnalyzer()
analyzer.create_gui()
