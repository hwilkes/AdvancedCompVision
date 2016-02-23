import numpy as np
import cv2
mouseone = cv2.imread('mouseone.jpg')
mousetwo = cv2.imread('mousetwo.jpg')

grayOne = cv2.cvtColor(mouseone, cv2.COLOR_BGR2GRAY)
momentsOne = cv2.moments(grayOne)
huMomentsOne = cv2.HuMoments(momentsOne)
print huMomentsOne[1]

grayTwo = cv2.cvtColor(mousetwo, cv2.COLOR_BGR2GRAY)
momentsTwo = cv2.moments(grayTwo)
huMomentsTwo = cv2.HuMoments(momentsTwo)

mouseone = cv2.copyMakeBorder(mouseone,0,50,0,0,cv2.BORDER_REPLICATE)
font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(mouseone,'M1: '+str(round(momentsOne['m01'],3))+"    M2: "+str(round(momentsOne['m02'],3))+"    M3: "+str(round(momentsOne['m03'],3))
            ,(10,500), font,0.40,(0,0,0),1,cv2.LINE_AA)
cv2.putText(mouseone,'MU1: '+str(round(huMomentsOne[0],3))+"    MU2: "+str(round(huMomentsOne[1],3))+"    MU3: "+str(round(huMomentsOne[2],3))
            ,(10,550), font,0.40,(0,0,0),1,cv2.LINE_AA)
cv2.imshow('mouse one',mouseone)

mousetwo = cv2.copyMakeBorder(mousetwo,0,50,0,0,cv2.BORDER_REPLICATE)
font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(mousetwo,'M1: '+str(round(momentsTwo['m01'],3))+"    M2: "+str(round(momentsTwo['m02'],3))+"    M3: "+str(round(momentsTwo['m03'],3))
            ,(10,450), font,0.40,(0,0,0),1,cv2.LINE_AA)
cv2.putText(mouseone,'MU1: '+str(round(huMomentsTwo[0],3))+"    MU2: "+str(round(huMomentsTwo[1],3))+"    MU3: "+str(round(huMomentsTwo[2],3))
            ,(10,500), font,0.40,(0,0,0),1,cv2.LINE_AA)
cv2.imshow('mouse two',mousetwo)

cv2.waitKey(0)
cv2.destroyAllWindows()
#cap = cv2.VideoCapture(0)

#while(True):
    # Capture frame-by-frame
 #   ret, frame = cap.read()
  #  gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
   # moments = cv2.moments(gray)
   # huMoments = cv2.HuMoments(moments)

    #ret,thresh = cv2.threshold(gray,127,255,0)
    #im2, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    #cv2.drawContours(frame, contours, -1, (0,255,0), 3)

    #cv2.imshow('frame',frame)
    #if cv2.waitKey(1) & 0xFF == ord('q'):
     #   break

# When everything done, release the capture
#cap.release()
#cv2.destroyAllWindows()