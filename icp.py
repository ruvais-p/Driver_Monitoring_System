import cv2
import imutils
from imutils import face_utils
import dlib
from scipy.spatial import distance
from pygame import mixer
import serial
import time
import customtkinter as ctk
from PIL import Image, ImageTk

# Initialize serial communication
serial_conn = serial.Serial('COM4', 115200)
time.sleep(1)

# Initialize pygame mixer
mixer.init()
mixer.music.load("music.wav")

def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

thresh = 0.25
flag = 0
frame_check = 20

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS['left_eye']
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS['right_eye']

detect = dlib.get_frontal_face_detector()
predict = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
fire_cascade = cv2.CascadeClassifier("fire_detection.xml")
cap = cv2.VideoCapture(0)

class GUI(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.geometry("1000x600")  # Adjusted width to accommodate the label
        self.title('Driver Monitoring System')
        self.resizable(False, False)
        
        # Define colors
        self.default_color = "#395894"
        self.alert_color = "#B52323"
        
        # Create a frame to hold the video feed and the white label
        self.main_frame = ctk.CTkFrame(self)
        self.main_frame.grid(row=0, column=0, sticky="nsew")
        
        self.label = ctk.CTkLabel(self.main_frame, text="", compound="center")
        self.label.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        
        # Create the white label on the right side
        self.side_label = ctk.CTkFrame(self, bg_color="white", width=200)
        self.side_label.grid(row=0, column=1, sticky="ns")
        
        # Configure grid for the side label
        self.side_label.grid_rowconfigure(0, weight=1)
        self.side_label.grid_rowconfigure(1, weight=1)
        self.side_label.grid_rowconfigure(2, weight=1)
        self.side_label.grid_rowconfigure(3, weight=1)
        self.side_label.grid_columnconfigure(0, weight=1)

        # Load images
        try:
            self.drowsy_image = ImageTk.PhotoImage(Image.open("witness.png").resize((50, 50)))
            self.fire_image = ImageTk.PhotoImage(Image.open("fire.png").resize((50, 50)))
        except Exception as e:
            print(f"Error loading images: {e}")
            self.drowsy_image = None
            self.fire_image = None
        
        # Create labels for images
        self.drowsy_image_label = ctk.CTkLabel(self.side_label, image=self.drowsy_image, bg_color=self.default_color, text='')
        self.drowsy_image_label.grid(row=0, pady=(10, 0), padx=(10,10), sticky="nsew")
        
        self.drowsy_text_label = ctk.CTkLabel(self.side_label, text="Not Drowsy",
                                              fg_color=self.default_color, corner_radius=0, padx=20, pady=10)
        self.drowsy_text_label.grid(row=1, pady=(0, 10), padx=(10,10), sticky="nsew")
        
        self.fire_image_label = ctk.CTkLabel(self.side_label, image=self.fire_image, bg_color=self.default_color, text='')
        self.fire_image_label.grid(row=2, pady=(10, 0), padx=(10,10), sticky="nsew")
        
        self.fire_text_label = ctk.CTkLabel(self.side_label, text="No Fire",
                                            fg_color=self.default_color, corner_radius=0, padx=20, pady=10)
        self.fire_text_label.grid(row=3, pady=(0, 20), padx=(10,10), sticky="nsew")
        
        self.update_frame()
        
    def update_frame(self):
        ret, frame = cap.read()
        if ret:
            # Resize frame to fit the window
            frame = imutils.resize(frame, width=self.winfo_width() - 200, height=self.winfo_height())
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            subjects = detect(gray, 0)

            # Initialize detection flags
            eye_detected = 0
            fire_detected = 0

            for subject in subjects:
                shape = predict(gray, subject)
                shape = face_utils.shape_to_np(shape)
                leftEye = shape[lStart:lEnd]
                rightEye = shape[rStart:rEnd]
                leftEar = eye_aspect_ratio(leftEye)
                rightEar = eye_aspect_ratio(rightEye)
                ear = (leftEar + rightEar) / 2.0
                leftEyeHull = cv2.convexHull(leftEye)
                rightEyeHull = cv2.convexHull(rightEye)
                cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
                cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

                if ear < thresh:
                    global flag
                    flag += 1
                    if flag >= frame_check:
                        cv2.putText(frame, "*ALERT!*", (500, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        cv2.putText(frame, "*ALERT!*", (500, 700),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        eye_detected = 1
                        if not mixer.music.get_busy():
                            mixer.music.play()
                else:
                    flag = 0

            # Fire Detection
            fire = fire_cascade.detectMultiScale(frame, 1.2, 5)
            for (x, y, w, h) in fire:
                cv2.rectangle(frame, (x-20, y-20), (x+w+20, y+h+20), (255, 0, 0), 2)
                cv2.putText(frame, "*FIRE DETECTED!*", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                fire_detected = 1

            # Update UI indicators
            self.drowsy_text_label.configure(text="Drowsy" if eye_detected else "Not Drowsy",
                                             fg_color=self.alert_color if eye_detected else self.default_color)
            self.fire_text_label.configure(text="Fire Detected" if fire_detected else "No Fire",
                                           fg_color=self.alert_color if fire_detected else self.default_color)
            self.drowsy_image_label.configure(bg_color=self.alert_color if eye_detected else self.default_color)
            self.fire_image_label.configure(bg_color=self.alert_color if fire_detected else self.default_color)

            # Construct array format data
            data_array = [str(eye_detected), str(fire_detected),]
            data_string = ','.join(data_array) + '\n'

            # Send data over serial connection
            serial_conn.write(data_string.encode())

            # Convert frame to ImageTk format
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            imgtk = ImageTk.PhotoImage(image=img)
            self.label.imgtk = imgtk
            self.label.configure(image=imgtk)
        
        self.after(10, self.update_frame)

if __name__ == '__main__':
    app = GUI()
    app.mainloop()

# Cleanup
cap.release()
cv2.destroyAllWindows()
