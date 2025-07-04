# Drowsiness-Detection

This project is a real-time drowsiness detection system built using Python, OpenCV, and dlib. It helps in monitoring driver fatigue by analyzing eye movement and triggering alerts when signs of drowsiness are detected. The system uses facial landmark detection to track the Eye Aspect Ratio (EAR) and determines if the eyes remain closed for a prolonged duration, indicating drowsiness.

# Download the Pre-trained Shape Predictor
The project requires shape_predictor_68_face_landmarks.dat, a pre-trained model for facial landmark detection.

🔗 Download it from here:
http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2

Then:
Extract the .bz2 file.
Place the shape_predictor_68_face_landmarks.dat file in the project directory (or specify its path in the script).


# Features

* Real-time webcam feed monitoring
* Eye Aspect Ratio (EAR) calculation for detecting eye closure
* Sound alarm to alert the user when drowsiness is detected
* Simple and lightweight implementation using Python and OpenCV

# Technologies Used

* Python
* OpenCV
* dlib
* imutils
* scipy

# How It Works

1. Captures live video from webcam.
2. Detects facial landmarks using `dlib`'s pre-trained shape predictor.
3. Calculates the Eye Aspect Ratio (EAR) to monitor eye closure.
4. If EAR falls below a threshold for a certain number of frames, it triggers a drowsiness alert

# Applications

* Driver safety and monitoring systems
* Workplace fatigue monitoring
* Personal alert systems for long computer usage

# Note

* Requires a webcam to run in real-time.
* Performance may vary based on lighting and camera quality.
