import cv2
import numpy as np
import pickle
import matplotlib.pyplot as plt

# parking space rectangle dimensions
rectW, rectH = 107, 48

# Load video and position data
cap = cv2.VideoCapture('CarPark1.mp4')
with open('CarParkPos', 'rb') as f:
    posList = pickle.load(f)

frame_counter = 0

# function to check parking spaces
def check(imgPro, img):
    spaceCount = 0
    for pos in posList:
        x, y = pos
        crop = imgPro[y:y + rectH, x:x + rectW]
        count = cv2.countNonZero(crop)
        if count < 900:
            color = (0, 255, 0)
            thickness = 5
            spaceCount += 1
        else:
            color = (0, 0, 255)
            thickness = 2
        cv2.rectangle(img, pos, (pos[0] + rectW, pos[1] + rectH), color, thickness)
    cv2.rectangle(img, (45, 30), (250, 75), (100, 0, 100), 1)
    cv2.putText(img, f'Free: {spaceCount}/{len(posList)}', (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

# Setup for showing frames
plt.ion()  # Enable interactive mode to update frames
fig, ax = plt.subplots()

# main loop
while True:
    ret, img = cap.read()
    if not ret:
        break

    frame_counter += 1
    if frame_counter == cap.get(cv2.CAP_PROP_FRAME_COUNT):
        frame_counter = 0
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 1)
    Thre = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY_INV, 25, 16)

    blur = cv2.medianBlur(Thre, 5)
    kernel = np.ones((3, 3), np.uint8)
    dilate = cv2.dilate(blur, kernel, iterations=1)

    check(dilate, img)

    # Display frame
    ax.clear()  # Clear the previous image
    ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    ax.axis('off')  # Hide the axis labels
    plt.draw()
    plt.pause(0.01)  # Pause to allow for the frame to update

# Close video and plot
cap.release()
plt.ioff()  # Disable interactive mode
plt.show()  # Make sure the last frame shows if paused
