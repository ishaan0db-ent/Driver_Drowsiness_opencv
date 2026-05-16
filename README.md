# Driver Drowsiness Detection System

This project is a real-time Driver Drowsiness Detection System developed using Computer Vision and AI-based techniques. The main objective of the project is to detect signs of driver fatigue by continuously monitoring eye movements through a webcam feed.

The system uses MediaPipe FaceMesh to detect facial landmarks and calculates the Eye Aspect Ratio (EAR) to determine whether the driver’s eyes are open or closed. If the eyes remain closed for a certain number of consecutive frames, the system identifies the driver as drowsy and generates an alert.

The project was built using Python, OpenCV, MediaPipe, and Streamlit. Streamlit was used to create an interactive user interface and deployment-ready application.

# Features

- Real-time webcam monitoring
- Eye tracking using MediaPipe FaceMesh
- Eye Aspect Ratio (EAR) based detection
- Drowsiness alert system
- Streamlit-based interface
- Optimized real-time performance

# Technologies Used

- Python
- OpenCV
- MediaPipe
- Streamlit
- NumPy
- Computer Vision

# Project Structure

driver_drowsiness_opencv/

├── app.py  
├── mediapipe_app.py  
├── requirements.txt  
├── runtime.txt  
├── README.md  
├── data/  
└── src/  
# Project Preview

## Main Interface
![Main UI](Image 1.jpeg)

## Real-Time Detection
![Detection](Image 2.jpeg)


# Installation

Clone the repository:

```bash
git clone https://github.com/your-username/Driver_Drowsiness_opencv.git