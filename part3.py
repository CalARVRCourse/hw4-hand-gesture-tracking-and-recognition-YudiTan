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
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)  
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU )
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)     
    contours=sorted(contours,key=cv2.contourArea,reverse=True)    
    fingerCount = 0  
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
 
                        cv2.line(thresh,start,end,[0,255,0],2)  
    cv2.imshow("Contours", thresh)
    print(fingerCount)
    cv2.waitKey(1)





cv2.destroyAllWindows() 
cam.release()