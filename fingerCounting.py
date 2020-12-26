#-----------------------------------------------------------------------------------------------------------------------------------------------
# Imports
import numpy as np
import cv2
import math
from pyautogui import press, typewrite, hotkey, keyDown, keyUp  # For controlling a media player(example VLC)
import time

#-----------------------------------------------------------------------------------------------------------------------------------------------
# Function for obtaining threshold image from the cropped RGB image of the hand
def obtainThresholdImage(crop_image):

    # Applying Gaussian blur (for smoothening)
    blur = cv2.GaussianBlur(crop_image, (3, 3), 0)
    # Changing color-space from BGR -> HSV (for skin color detection)
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    # Creating a binary image with where the skin is represented by white and rest is black
    mask2 = cv2.inRange(hsv, np.array([2, 0, 0]), np.array([20, 255, 255]))
    # Kernel for morphological transformation (5x5)
    kernel = np.ones((5, 5))
    # Applying morphological transformations to filter out the background noise (dilation and then erosion)
    dilated = cv2.dilate(mask2, kernel, iterations=1)
    eroded = cv2.erode(dilated, kernel, iterations=1)
    # Applying Gaussian Blur and Thresholding
    filtered = cv2.GaussianBlur(eroded, (3, 3), 0)
    ret, thresh = cv2.threshold(filtered, 127, 255, 0)

    return thresh
#-----------------------------------------------------------------------------------------------------------------------------------------------
# Function for Gesture commands using PyAutoGUI

def gestureControl(count_defects, last_recorded_time):

    coolDown = 2.0 # 2 sec cool down between two commands
    current_time = time.time() # recording current time
    if current_time - last_recorded_time >= coolDown:
        if count_defects == 0: #PAUSE
            press('space') 
        elif count_defects == 1: #PLAY
            press('space') 
        elif count_defects == 2: #FORWARD
            press('right')    
        elif count_defects == 3: #BACKWARD
            press('left')
        elif count_defects == 4: #SOME OTHER FUNCTIONALITY CAN BE ADDED
            pass
        else:
            pass
        last_recorded_time = current_time
    else:
        pass
    return last_recorded_time

#-----------------------------------------------------------------------------------------------------------------------------------------------
# Function for counting the number of Defects in the convex hull of the thresholded hand gesture image

def countDefects(defects, contour, crop_image):
    # Using the cosine rule to find angle of the far point from the start and end point i.e. the convex points (representing
    # the finger tips) for all the defects
    count_defects = 0

    for i in range(defects.shape[0]):
        s, e, f, d = defects[i, 0]
        start = tuple(contour[s][0])
        end = tuple(contour[e][0])
        far = tuple(contour[f][0])

        a = math.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
        b = math.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
        c = math.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)
        angle = (math.acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c)) * 180) / 3.14

        # if the angle > 90, draw a circle at the far point
        if angle <= 90:
            count_defects += 1
            cv2.circle(crop_image, far, 1, [0, 0, 255], -1)

        cv2.line(crop_image, start, end, [0, 255, 0], 2)

    return count_defects

#-----------------------------------------------------------------------------------------------------------------------------------------------
# Function to print the number of defects and the corresponding command gesture

def printDefects(count_defects, frame):
    if count_defects == 0:
            cv2.putText(frame, "ONE (PLAY/PAUSE)", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2,(255,0,0),2)
    elif count_defects == 1:
        cv2.putText(frame, "TWO (PLAY/PAUSE)", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2,(255,0,0), 2)
    elif count_defects == 2:
        cv2.putText(frame, "THREE (FORWARD)", (5, 50), cv2.FONT_HERSHEY_SIMPLEX, 2,(255,0,0), 2)
    elif count_defects == 3:
        cv2.putText(frame, "FOUR (BACKWARD)", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2,(255,0,0), 2)
    elif count_defects == 4:
        cv2.putText(frame, "FIVE", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2,(255,0,0), 2)
    else:
        pass

#-----------------------------------------------------------------------------------------------------------------------------------------------
# Driver code

# Open the Camera

# capture = cv2.VideoCapture('https://192.168.43.1:8080/video') # using IP camera (of phone) via local server

capture = cv2.VideoCapture(0) # Webcam can also be used but as stated, phone camera provides better contrast                                                               
                                                            # (phone camera for better resolution and contrast)

# Keeping track of the time to avoid layering up of commands
last_recorded_time = time.time()       

while capture.isOpened():

    # Capturing the frames from the camera
    ret, frame = capture.read()

    # Getting the hand gesture image data from a (300x300) bounding rectangle with bounding box(100, 400, 100, 400) 
    cv2.rectangle(frame, (100, 100), (400, 400), (0, 255, 0), 0)
    crop_image = frame[100:400, 100:400]
    
    # Getting the Threshold image from the crop_image
    thresh = obtainThresholdImage(crop_image)

    # Showing the threshold image (BW)
    cv2.imshow("Thresholded", thresh)

    # Finding the contours
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    try:
        # Finding the contour with maximum area (leaving out the noisy contours with small areas, if any)
        contour = max(contours, key=lambda x: cv2.contourArea(x))

        # Creating bounding rectangle around the hand contour (area in which the hand resides)
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(crop_image, (x, y), (x + w, y + h), (0, 0, 255), 0)

        # Finding the convex hull
        hull = cv2.convexHull(contour)

        # Drawing contour around the detected hand
        drawing = np.zeros(crop_image.shape, np.uint8)
        cv2.drawContours(drawing, [contour], -1, (0, 255, 0), 0)
        cv2.drawContours(drawing, [hull], -1, (0, 0, 255), 0)

        # Finding the convexity defects
        hull = cv2.convexHull(contour, returnPoints=False)
        defects = cv2.convexityDefects(contour, hull)

        # Calling the function to count the number of defects (related to the number of fingers raised)
        count_defects = countDefects(defects, contour, crop_image)

        # Printing the number of fingers in the hand gesture
        printDefects(count_defects, frame)

        #GESTURE CONTROL based on number of fingers raised
        last_recorded_time = gestureControl(count_defects, last_recorded_time)

    except:
        pass

    # Showining required image windows
    cv2.imshow("Gesture", frame)
    all_image = np.hstack((drawing, crop_image))
    cv2.imshow('Contours', all_image)

    # Close the camera if 'q' is pressed
    if cv2.waitKey(1) == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()
#----------------------------------------------------------------------------------------------------------------------------------------------

