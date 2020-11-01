import numpy as np
import cv2
import pyautogui

def skinmask(frame):
    lower_HSV = np.array([0, 40, 0], dtype = "uint8")
    upper_HSV = np.array([25, 255, 255], dtype = "uint8")
  
    convertedHSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    skinMaskHSV = cv2.inRange(convertedHSV, lower_HSV, upper_HSV)
  
  
    lower_YCrCb = np.array((0, 138, 67), dtype = "uint8")
    upper_YCrCb = np.array((255, 173, 133), dtype = "uint8")
      
    convertedYCrCb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    skinMaskYCrCb = cv2.inRange(convertedYCrCb, lower_YCrCb, upper_YCrCb)
  
    skinMask = cv2.add(skinMaskHSV,skinMaskYCrCb)
    
    return skinMask

def denoisemask(frame, skinMask):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    skinMask = cv2.erode(skinMask, kernel, iterations = 2)
    skinMask = cv2.dilate(skinMask, kernel, iterations = 2)
  
    # blur the mask to help remove noise, then apply the
    # mask to the frame
    skinMask = cv2.GaussianBlur(skinMask, (7, 7), 0)
    skin = cv2.bitwise_and(frame, frame, mask = skinMask)
    return skin

def connectedComponentAnalysis(frame):
    #binarize the image
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    max_binary_value = 255
    ret, thresh = cv2.threshold(gray, 0, max_binary_value, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU )
    
    #connected components analysis
    ret, markers, stats, centroids = cv2.connectedComponentsWithStats(thresh, ltype=cv2.CV_16U)

    #find and isolate the ring
    flag = False
    bimanualDistance = 0
    cnt = 0
    roi = 0
    if (ret>3):
        try:
            centroidsSortedByArea = centroids[np.argsort(stats[:, 4])]
            roi = centroidsSortedByArea[-3:-1,::]
            x1 = roi[0,0]
            y1 = roi[0,1]
            x2 = roi[1,0]
            y2 = roi[1,1]
            bimanualDistance = np.sqrt((x1-x2)**2+(y1-y2)**2)
            
            flag = True
            
        except:
            print("No hands found")
            flag = False

    return ret&flag, bimanualDistance
    
def bimanualGesture():
    # set up the camera
    cam = cv2.VideoCapture(0)
    cam.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25) # 0.25 turns OFF auto exp
    cam.set(cv2.CAP_PROP_AUTO_WB, 0.25) # 0.25 turns OFF auto WB
    
    bimanualDistance_prev = 0;
    while True:
        ret, frame = cam.read()
        # display the current image
        cv2.imshow("frame (press q to exit)", frame)
        k = cv2.waitKey(1) #k is the key pressed
        if k == 27 or k==113:  #27, 113 are ascii for escape and q respectively
            cv2.destroyAllWindows()
            cam.release()
            return
        
        skinMask = skinmask(frame)
        skin = denoisemask(frame, skinMask)
        flag, bimanualDistance = connectedComponentAnalysis(skin)
        if flag:
            if bimanualDistance_prev==0: bimanualDistance_prev = bimanualDistance
            if bimanualDistance > 1.2*bimanualDistance_prev:
                bimanualDistance_prev = bimanualDistance
                pyautogui.hotkey('command', '+')
            if bimanualDistance < bimanualDistance_prev/1.2:
                bimanualDistance_prev = bimanualDistance
                pyautogui.hotkey('command', '-')


bimanualGesture()
