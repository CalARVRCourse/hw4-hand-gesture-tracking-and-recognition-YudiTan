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
    ret, markers, stats, centroids = cv2.connectedComponentsWithStats(thresh,ltype=cv2.CV_16U)  
    markers = np.array(markers, dtype=np.uint8)  
    label_hue = np.uint8(179*markers/np.max(markers))  
    blank_ch = 255*np.ones_like(label_hue)  
    labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])
    labeled_img = cv2.cvtColor(labeled_img,cv2.COLOR_HSV2BGR)
    labeled_img[label_hue==0] = 0  
    statsSortedByArea = stats[np.argsort(stats[:, 4])]  

    if (ret>2):  
        try:  
            roi = statsSortedByArea[-3][0:4]  
            x, y, w, h = roi  
            subImg = labeled_img[y:y+h, x:x+w]  
            subImg = cv2.cvtColor(subImg, cv2.COLOR_BGR2GRAY);  
            contours, _ = cv2.findContours(subImg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  
            maxCntLength = 0  
            for i in range(0,len(contours)):  
                cntLength = len(contours[i])  
                if(cntLength>maxCntLength):  
                    cnt = contours[i]  
                    maxCntLength = cntLength  
            if(maxCntLength>=5):  
                ellipseParam = cv2.fitEllipse(cnt)  
                subImg = cv2.cvtColor(subImg, cv2.COLOR_GRAY2RGB);  
                subImg = cv2.ellipse(subImg,ellipseParam,(0,255,0),2)  
                
            subImg = cv2.resize(subImg, (0,0), fx=3, fy=3)  
            (x,y),(MA,ma),angle = cv2.fitEllipse(cnt)
            print(x, y, MA, ma)
            cv2.imshow("ROI "+str(2), subImg)  
            cv2.waitKey(0)  
        except:  
            print("No hand found")  


    # cv2.imshow("Threshold", labeled_img)  
    # cv2.waitKey(1) 
    # ret, markers, stats, centroids = cv2.connectedComponentsWithStats(frame,ltype=cv2.CV_16U) 
    # if not ret:
    #     print("Unable to find connected components")
    #     break
    # markers = np.array(markers, dtype=np.uint8)  
    # label_hue = np.uint8(179*markers/np.max(markers))  
    # blank_ch = 255*np.ones_like(label_hue)  
    # labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])
    # labeled_img = cv2.cvtColor(labeled_img,cv2.COLOR_HSV2BGR)
    # labeled_img[label_hue==0] = 0  
    # statsSortedByArea = stats[np.argsort(stats[:, 4])]  
    # if (ret>2):  
    #     try:  
    #         roi = statsSortedByArea[-3][0:4]  
    #         x, y, w, h = roi  
    #         subImg = labeled_img[y:y+h, x:x+w]  
    #         subImg = cv2.cvtColor(subImg, cv2.COLOR_BGR2GRAY);  
    #         _, contours, _ = cv2.findContours(subImg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  
    #         maxCntLength = 0  
    #         for i in range(0,len(contours)):  
    #             cntLength = len(contours[i])  
    #             if(cntLength>maxCntLength):  
    #                 cnt = contours[i]  
    #                 maxCntLength = cntLength  
    #         if(maxCntLength>=5):  
    #             ellipseParam = cv2.fitEllipse(cnt)  
    #             subImg = cv2.cvtColor(subImg, cv2.COLOR_GRAY2RGB);  
    #             subImg = cv2.ellipse(subImg,ellipseParam,(0,255,0),2)  
              
    #         subImg = cv2.resize(subImg, (0,0), fx=3, fy=3)  
    #         cv2.imshow("ROI "+str(2), subImg)  
    #         cv2.waitKey(1)  
    #     except:  
    #         print("No hand found") 




cv2.destroyAllWindows() 
cam.release()