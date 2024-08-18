import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
import pyttsx3
import tkinter as tk
from PIL import ImageTk, Image

# Initialize OpenCV and HandDetector
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
classifier = Classifier(r"C:\Users\Melissa\AppData\Local\Programs\Python\Python311\Sign_language to voice\keras_model.h5",
                        r"C:\Users\Melissa\AppData\Local\Programs\Python\Python311\Sign_language to voice\labels.txt")
offset = 20
imgSize = 300
counter = 0
text_speech = pyttsx3.init()

labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'space', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

answer = ""  # Initialize answer variable

# Function to update camera feed
def update_camera():
    global answer
    success, img = cap.read()
    if not success:
        return

    imgOutput = img.copy()
    hands, img = detector.findHands(img)
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        # Ensure the cropping coordinates are within the image bounds
        y_start = max(0, y - offset)
        y_end = min(img.shape[0], y + h + offset)
        x_start = max(0, x - offset)
        x_end = min(img.shape[1], x + w + offset)

        imgCrop = img[y_start:y_end, x_start:x_end]

        if imgCrop.size > 0:  # Proceed only if imgCrop is valid
            imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255

            aspectRatio = h / w

            if aspectRatio > 1:
                k = imgSize / h
                wCal = math.ceil(k * w)
                imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                wGap = math.ceil((imgSize - wCal) / 2)
                imgWhite[:, wGap:wCal + wGap] = imgResize
                prediction, index = classifier.getPrediction(imgWhite, draw=False)
            else:
                k = imgSize / w
                hCal = math.ceil(k * h)
                imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                hGap = math.ceil((imgSize - hCal) / 2)
                imgWhite[hGap:hCal + hGap, :] = imgResize
                prediction, index = classifier.getPrediction(imgWhite, draw=False)

            if 0 <= index < len(labels):  # Ensure the index is within the valid range
                cv2.rectangle(imgOutput, (x - offset, y - offset - 70), (x - offset + 400, y - offset + 60 - 50), (0, 255, 0), cv2.FILLED)
                cv2.putText(imgOutput, labels[index], (x, y - 30), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 0), 2)
                cv2.rectangle(imgOutput, (x - offset, y - offset), (x + w + offset, y + h + offset), (0, 255, 0), 4)
                answer = labels[index]  # Update the answer with the detected label
            else:
                answer = ""  # Reset the answer if index is invalid

    imgRGB = cv2.cvtColor(imgOutput, cv2.COLOR_BGR2RGB)
    imgPIL = Image.fromarray(imgRGB)
    imgTK = ImageTk.PhotoImage(image=imgPIL)
    camera_label.imgTK = imgTK
    camera_label.config(image=imgTK)

    root.after(10, update_camera)

# Function to simulate capturing an image
def start_recognition():
    current_text = text_entry.get()
    if answer == 'space':
        text_entry.insert(tk.END, " ")
    else:
        text_entry.insert(tk.END, answer)

def play_sound():
    full_text = text_entry.get()
    text_speech.say(full_text)
    text_speech.runAndWait()

# Function to handle button hover enter
def on_enter(event):
    event.widget.config(bg="#FFA07A")  # Change background color on hover

# Function to handle button hover leave
def on_leave(event):
    event.widget.config(bg=event.widget.default_bg)  # Restore default background color on leave

# Function to close the window on 'Esc' key press
def close_window(event):
    root.destroy()

# Function to clear the last character in the text entry
def clear_last_character():
    current_text = text_entry.get()
    if current_text:  # Check if there is any text to delete
        text_entry.delete(len(current_text)-1)

# Create the main window
root = tk.Tk()
root.title("Camera UI")
root.attributes('-fullscreen', True)  # Set the window to full screen

# Bind the 'Esc' key to close the window
root.bind('<Escape>', close_window)

# Load background image
background_image = Image.open("bgscreen.jpg")  # Replace with your image file path
background_photo = ImageTk.PhotoImage(background_image)

# Create a label for the background image
background_label = tk.Label(root, image=background_photo)
background_label.place(x=0, y=0, relwidth=1, relheight=1)  # Use place to position the label over the entire window

# Centering the camera frame (simulated with a Label)
camera_label = tk.Label(root, bg="#F5F5DC", borderwidth=3, relief="solid")
camera_label.grid(row=1, column=0, columnspan=4, padx=30, pady=10, sticky=tk.NSEW)  # Use sticky to center

# Title label with bolder font, merged with background
title_label = tk.Label(root, text="VaaniConnect", font=('Arial', 50, 'bold'), fg='#6C3483', bg="#FDF5E6")
title_label.grid(row=0, column=0, columnspan=4, pady=(10, 20))  # Span across both columns at the top, with padding

# Animate the title label color (example of color cycling)
colors = ['#6C3483', '#2E86C1', '#27AE60', '#AF601A']  # List of darker colors to cycle through
def animate_title_color(index=0):
    next_index = (index + 1) % len(colors)
    next_color = colors[next_index]
    title_label.config(fg=next_color)
    root.after(1000, animate_title_color, next_index)  # Repeat every 1000 milliseconds (1 second)
animate_title_color()

root.rowconfigure(0, weight=1)  # Make row 0 expandable
root.rowconfigure(1, weight=1)  # Make row 1 expandable
root.rowconfigure(2, weight=1)  # Make row 2 expandable
root.columnconfigure(0, weight=1)  # Make column 0 expandable
root.columnconfigure(1, weight=1)  # Make column 1 expandable
root.columnconfigure(2, weight=1)  # Make column 2 expandable
root.columnconfigure(3, weight=1)  # Make column 3 expandable

text_entry = tk.Entry(root, width=20, font=('Arial', 40), borderwidth=3, relief="sunken")  # Adjust the width and font size as needed
text_entry.grid(row=2, column=0, columnspan=4, padx=10, pady=(20, 10), sticky=tk.NSEW)  # Span across both columns with extra padding at the top

# Start recognition button with peach color
recognition_button = tk.Button(root, text="START RECOGNITION", width=20, height=3, bg="#8B8378", fg="white", font=('serif', 10, 'bold'), command=start_recognition)
recognition_button.grid(row=3, column=1, padx=10, pady=10, sticky=tk.E)  # Align to the right (East)
recognition_button.default_bg = recognition_button['bg']  # Store default background color
recognition_button.bind('<Enter>', on_enter)  # Bind hover enter event
recognition_button.bind('<Leave>', on_leave)  # Bind hover leave event

# Audio button with peach color
audio_button = tk.Button(root, text="PLAY AUDIO", width=20, height=3, bg="#8B7355", fg="white", font=('serif', 10, 'bold'), command=play_sound)
audio_button.grid(row=3, column=2, padx=10, pady=10, sticky=tk.W)  # Align to the left (West)
audio_button.default_bg = audio_button['bg']  # Store default background color
audio_button.bind('<Enter>', on_enter)  # Bind hover enter event
audio_button.bind('<Leave>', on_leave)  # Bind hover leave event

# Clear button with hover effect and fixed width
clear_button = tk.Button(root, text="CLEAR LAST CHARACTER", width=25, height=3, bg="#F4A460", fg="white", font=('serif', 10, 'bold'), command=clear_last_character)
clear_button.grid(row=3, column=3, padx=10, pady=10)  # Align to the center (sticky=tk.NSEW for centering)
clear_button.default_bg = clear_button['bg']  # Store default background color
clear_button.bind('<Enter>', on_enter)  # Bind hover enter event
clear_button.bind('<Leave>', on_leave)  # Bind hover leave event

# Update the camera feed
update_camera()

# Start the GUI event loop
root.mainloop()

# Release the camera and close any OpenCV windows
cap.release()
cv2.destroyAllWindows()
