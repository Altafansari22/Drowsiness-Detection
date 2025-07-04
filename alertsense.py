import cv2
import dlib
import face_recognition
import numpy as np
from scipy.spatial import distance as dist
from tkinter import Tk, Label, Button, Canvas, PhotoImage
import threading
import pyttsx3
import time

# Global variables
running = False
alarm_enabled = True
alert_triggered = False
person_name = "Unknown"
last_called_name = ""
last_wake_up_time = 0

# Text-to-Speech Engine
engine = pyttsx3.init()

# Load known faces and names
known_face_encodings = []
known_face_names = []

def load_known_faces():
    global known_face_encodings, known_face_names
    image_files = [
        ("person1.jpg", "Altaf"),
        #("person2.jpg", "Jane Doe")
    ]
    for file, name in image_files:
        try:
            image = face_recognition.load_image_file(file)
            encoding = face_recognition.face_encodings(image)[0]
            known_face_encodings.append(encoding)
            known_face_names.append(name)
            print(f"Loaded face for {name}")
        except Exception as e:
            print(f"Error loading {file}: {e}")

# EAR Calculation
def calculate_ear(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

# Text-to-Speech for Wake Up Alert
def wake_up_alert(name):
    global last_wake_up_time
    current_time = time.time()
    if current_time - last_wake_up_time > 10:  # Prevent frequent calls
        print(f"Wake up alert for {name}")
        engine.say(f"Wake up {name}, you are drowsy!")
        engine.runAndWait()
        last_wake_up_time = current_time

# Main Detection Function
def detect_drowsiness(status_label, name_label):
    global running, alert_triggered, person_name, last_called_name
    running = True

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    (left_start, left_end) = (42, 48)
    (right_start, right_end) = (36, 42)
    EAR_THRESHOLD = 0.25
    EAR_CONSEC_FRAMES = 20
    frame_count = 0
    face_recog_skip_frames = 10
    face_recog_counter = 0

    cap = cv2.VideoCapture(0)

    while running:
        ret, frame = cap.read()
        if not ret:
            break

        small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        # Face Recognition
        if face_recog_counter % face_recog_skip_frames == 0:
            face_encodings = face_recognition.face_encodings(rgb_small_frame)
            for face_encoding in face_encodings:
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances) if matches else None
                if best_match_index is not None and matches[best_match_index]:
                    person_name = known_face_names[best_match_index]
                else:
                    person_name = "Unknown"
            name_label.config(text=f"Identified: {person_name}")

        face_recog_counter += 1
        faces = detector(gray)
        for face in faces:
            landmarks = predictor(gray, face)
            left_eye = [landmarks.part(i) for i in range(left_start, left_end)]
            right_eye = [landmarks.part(i) for i in range(right_start, right_end)]
            left_eye_coords = [(point.x, point.y) for point in left_eye]
            right_eye_coords = [(point.x, point.y) for point in right_eye]
            ear = (calculate_ear(left_eye_coords) + calculate_ear(right_eye_coords)) / 2.0

            if ear < EAR_THRESHOLD:
                frame_count += 1
                if frame_count >= EAR_CONSEC_FRAMES and not alert_triggered:
                    wake_up_alert(person_name)
                    alert_triggered = True
                    status_label.config(text=f"⚠ Wake Up {person_name}! You are Drowsy! ⚠", fg="red")
            else:
                frame_count = 0
                alert_triggered = False
                status_label.config(text="Monitoring...", fg="green")

        cv2.imshow("Driver Drowsiness Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Start and Stop Detection Functions
def start_detection(status_label, name_label):
    threading.Thread(target=detect_drowsiness, args=(status_label, name_label)).start()

def stop_detection():
    global running
    running = False

# Beautified GUI Setup
def setup_gui():
    root = Tk()
    root.title("Driver Drowsiness Detection")
    root.geometry("500x400")
    root.configure(bg="#e6f7ff")

    # Heading
    Label(root, text="Driver Drowsiness Detection", font=("Helvetica", 20, "bold"), fg="#003366", bg="#e6f7ff").pack(pady=10)

    # Canvas for logo or design
    #canvas = Canvas(root, width=100, height=100, bg="#e6f7ff", highlightthickness=0)
    #canvas.pack()
    #img = PhotoImage(file="icon.jpg")  # Use your logo file path
    #canvas.create_image(50, 50, image=img)

    # Labels for status and identified person
    status_label = Label(root, text="Monitoring...", font=("Arial", 14), fg="green", bg="#e6f7ff")
    status_label.pack(pady=10)

    name_label = Label(root, text="Identified: Unknown", font=("Arial", 14), fg="#003366", bg="#e6f7ff")
    name_label.pack(pady=5)

    # Start and Stop Buttons
    Button(root, text="Start Detection", font=("Arial", 14), bg="#28a745", fg="white",
           command=lambda: start_detection(status_label, name_label)).pack(pady=10)

    Button(root, text="Stop Detection", font=("Arial", 14), bg="#dc3545", fg="white",
           command=stop_detection).pack(pady=5)

    # Footer
    Label(root, text="Stay Safe on the Road!", font=("Arial", 12, "italic"), fg="#003366", bg="#e6f7ff").pack(side="bottom", pady=10)

    root.mainloop()

if __name__ == "__main__":
    load_known_faces()
    setup_gui()
