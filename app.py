import cv2
import time
import glob
import os
import streamlit as st
from emailing import send_email
from threading import Thread

# Streamlit UI
st.set_page_config(page_title= "Motion Detector", layout="centered")
st.title("Motion Detection Web App")
start_button = st.button("Start Camera")
stop_button = st.button("Stop Camera")

if 'run' not in st.session_state:
    st.session_state.run = False

if start_button:
    st.session_state.run = True
if stop_button:
    st.session_state.run = False

# Placeholder for camera feed
frame_placeholder = st.empty()

# Define the cleanup function
def clear_images_folder():
    images = glob.glob("images/*.png")
    for image in images:
        os.remove(image)

# If user starts the camera
if st.session_state.run:
    # Capturing the Video through the webcam
    video = cv2.VideoCapture(0)
    time.sleep(1)

    first_frame = None
    status_list = []
    count = 1

    while st.session_state.run:
        status = 0

        check, frame = video.read()
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_frame_gau = cv2.GaussianBlur(gray_frame, (21, 21), 0)

        if first_frame is None:
            first_frame = gray_frame_gau

        delta_frame = cv2.absdiff(first_frame, gray_frame_gau)
        thres_frame = cv2.threshold(delta_frame, 30, 255, cv2.THRESH_BINARY)[1]
        dil_frame = cv2.dilate(thres_frame, None, iterations=2)

        contours, check = cv2.findContours(dil_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            if cv2.contourArea(contour) < 5000:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            rectangle = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)

            if rectangle.any():
                status = 1
                cv2.imwrite(f"images/{count}.png", frame)
                count += 1
                all_images = glob.glob("images/*.png")
                index = int(len(all_images) / 2)
                image_with_object = all_images[index]

        status_list.append(status)
        status_list = status_list[-2:]

        if status_list[0] == 1 and status_list[1] == 0:
            email_thread = Thread(target=send_email, args=(image_with_object,))
            email_thread.daemon = True
            cleaning_thread = Thread(target=clear_images_folder)
            cleaning_thread.daemon = True

            email_thread.start()

        # Display the frame in Streamlit
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(frame_rgb, channels="RGB")

        # Exit condition
        if not st.session_state.run:
            cleaning_thread.start()
            break

    video.release()
