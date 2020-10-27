import cv2
import numpy as np

# setting up video capture
cam = cv2.VideoCapture(0)

lower_HSV = np.array([0, 40, 0], dtype = "uint8")  
upper_HSV = np.array([25, 255, 255], dtype = "uint8")  




lower_YCrCb = np.array((0, 138, 67), dtype = "uint8")  
upper_YCrCb = np.array((255, 173, 133), dtype = "uint8")  



def update_(x):  
    pass 

while True:
    ret, frame = cam.read()
    if not ret:
        print("Unable to capture video")
        break

    # part 1 
    cv2.namedWindow("part_1")    
    # Create a Trackbar to choose a value for a parameter    
    # cv2.createTrackbar(parameter_value_name, window_name , parameter_min_value, parameter_max_value, nothing)  

    convertedHSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)  
    skinMaskHSV = cv2.inRange(convertedHSV, lower_HSV, upper_HSV)  

        
    convertedYCrCb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)  
    skinMaskYCrCb = cv2.inRange(convertedYCrCb, lower_YCrCb, upper_YCrCb)  

    skinMask = cv2.add(skinMaskHSV,skinMaskYCrCb)  

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))  
    skinMask = cv2.erode(skinMask, kernel, iterations = 2)  
    skinMask = cv2.dilate(skinMask, kernel, iterations = 2)  
    
    # blur the mask to help remove noise, then apply the  
    # mask to the frame  
    skinMask = cv2.GaussianBlur(skinMask, (3, 3), 0) 
    skin = cv2.bitwise_and(frame, frame, mask = skinMask) 
    cv2.imshow('frame', skin)
    k = cv2.waitKey(1)


cv2.destroyAllWindows() 
cam.release()