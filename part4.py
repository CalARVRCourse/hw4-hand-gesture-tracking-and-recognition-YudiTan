import cv2
import numpy as np
import pyautogui

# setting up video capture
cam = cv2.VideoCapture(0)
cam.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25) # 0.25 turns OFF auto exp
cam.set(cv2.CAP_PROP_AUTO_WB, 0.25) # 0.25 turns OFF auto WB

while True:
    lower_HSV = np.array([0, 48, 80], dtype = "uint8")  
    upper_HSV = np.array([25, 255, 255], dtype = "uint8")  
    lower_YCrCb = np.array((0, 138, 67), dtype = "uint8")  
    upper_YCrCb = np.array((255, 173, 133), dtype = "uint8")  
    ret, frame = cam.read()
    if not ret:
        print("Unable to capture video")
        break

    # part 1 
    convertedHSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)  
    skinMaskHSV = cv2.inRange(convertedHSV, lower_HSV, upper_HSV)  

        
    convertedYCrCb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)  
    skinMaskYCrCb = cv2.inRange(convertedYCrCb, lower_YCrCb, upper_YCrCb)  

    skinMask = cv2.add(skinMaskHSV,skinMaskYCrCb)  

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))  
    skinMask = cv2.erode(skinMask, kernel, iterations = 2)  
    skinMask = cv2.dilate(skinMask, kernel, iterations = 2)  
    
    # blur the mask to help remove noise, then apply the  
    # mask to the frame  
    skinMask = cv2.GaussianBlur(skinMask, (3, 3), 0) 
    skin = cv2.bitwise_and(frame, frame, mask = skinMask) 

    # part 2
    gray = cv2.cvtColor(skin,cv2.COLOR_BGR2GRAY)  
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)  
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)       
    contours=sorted(contours,key=cv2.contourArea,reverse=True)       
    thresh = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
    fingerCount = 1
    if len(contours)>1:  
        largestContour = contours[0]  
        hull = cv2.convexHull(largestContour, returnPoints = False)     
        for cnt in contours[:1]:  
            defects = cv2.convexityDefects(cnt,hull)  
            if(not isinstance(defects,type(None))):  
                for i in range(defects.shape[0]):  
                    s,e,f,d = defects[i,0]  
                    start = tuple(cnt[s][0])  
                    end = tuple(cnt[e][0])  
                    far = tuple(cnt[f][0])  
                            
                    c_squared = (end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2  
                    a_squared = (far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2  
                    b_squared = (end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2  
                    angle = np.arccos((a_squared + b_squared  - c_squared ) / (2 * np.sqrt(a_squared * b_squared )))    
                                        
                    if angle <= np.pi / 3:  
                        fingerCount += 1  
                        cv2.circle(thresh, far, 4, [0, 0, 255], -1)
                    thresh = cv2.line(thresh,start,end,(0,255,0),2)  
        M = cv2.moments(largestContour)  

        # laptop resolution: 3072 * 1920
        # detection window resolution: 1080 * 1920
        scaleX = 3072 / 1080
        scaleY = 1920 / 1920
        offsetX = 0
        offsetY = 0
        cX = offsetX + scaleX *int(M["m10"] / M["m00"])  
        cY = offsetY + scaleY *int(M["m01"] / M["m00"])  
        pyautogui.moveTo(cX, cY, duration=0.02, tween=pyautogui.easeInOutQuad) 

        # print("Finger count: ", fingerCount)
        cv2.imshow("Contour: ", thresh)
        cv2.waitKey(1)

   



cv2.destroyAllWindows() 
cam.release()
