# Driver Monitoring System

## Overview

The **Driver Monitoring System** is a real-time application that detects drowsiness and fire hazards using computer vision and machine learning techniques. It monitors the driver's state through a webcam feed and alerts the driver when drowsiness is detected or if a fire is spotted in the surroundings. The system sends detection data through a serial connection to other devices for additional actions, such as controlling vehicle systems or notifying authorities.

The system is built using Python with libraries such as `OpenCV`, `dlib`, `imutils`, `scipy`, `pygame`, and `customtkinter`.

---

## Features

- **Drowsiness Detection**:
  - Monitors the driver's eyes using a pre-trained facial landmark model.
  - Calculates the Eye Aspect Ratio (EAR) to detect when the driver's eyes are closed for a prolonged period, indicating drowsiness.
  - Alerts the driver with both a visual warning and an audio alarm when drowsiness is detected.

- **Fire Detection**:
  - Detects fire in the video feed using a pre-trained Haar Cascade model for fire detection.
  - Visually marks detected fire on the video feed and updates the user interface (UI) with an alert message.

- **Serial Communication**:
  - Sends detection data (drowsiness and fire status) over a serial connection to other devices for further action.

- **Graphical User Interface (GUI)**:
  - Displays the live video feed and provides real-time status updates on drowsiness and fire detection.
  - Includes visual indicators (red for alert) to notify the user of drowsiness or fire hazards.

---

## Dependencies

The following Python libraries are required to run the Driver Monitoring System:

- `OpenCV`: For video processing and fire detection.
- `imutils`: For image processing and resizing.
- `dlib`: For facial landmark detection (68-point model).
- `scipy`: For calculating the Eye Aspect Ratio (EAR) used in drowsiness detection.
- `pygame`: For playing audio alerts.
- `customtkinter`: For building the graphical user interface (GUI).
- `Pillow`: For handling image formats within the GUI.
- `pyserial`: For serial communication with other devices.

To install the necessary dependencies, run:

```bash
pip install opencv-python opencv-python-headless imutils dlib scipy pygame customtkinter pillow pyserial
