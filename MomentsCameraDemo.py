import numpy as np
import cv2

cap = cv2.VideoCapture(0)


while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    moments = cv2.moments(gray)
    huMoments = cv2.HuMoments(moments)
    white = cv2.imread('whitespace.bmp')

    #ret,thresh = cv2.threshold(gray,127,255,0)
    #im2, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

   # cv2.drawContours(frame, contours, -1, (0,255,0), 3)

    #frame = cv2.copyMakeBorder(frame,0,100,0,0,cv2.BORDER_REPLICATE)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(white,'M1: '+str(round(moments['m01'],3))+"    M2: "+str(round(moments['m02'],3))+"    M3: "+str(round(moments['m03'],3))
            ,(10,50), font,0.40,(0,0,0),1,cv2.LINE_AA)
    cv2.putText(white,'MU1: '+str(round(huMoments[0],14))+"    MU2: "+str(round(huMoments[1],14))+"    MU3: "+str(round(huMoments[2],14))
            ,(10,100), font,0.40,(0,0,0),1,cv2.LINE_AA)
    cv2.putText(white,'MU4: '+str(round(huMoments[3],14))
            ,(10,150), font,0.40,(0,0,0),1,cv2.LINE_AA)
    cv2.imshow('output',white)
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()