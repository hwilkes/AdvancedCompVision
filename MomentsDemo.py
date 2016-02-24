import numpy as np
import cv2
mouseone = cv2.imread('mouseone.jpg')
mousetwo = cv2.imread('mousetwo.jpg')

grayOne = cv2.cvtColor(mouseone, cv2.COLOR_BGR2GRAY)

rows,cols = grayOne.shape
M = np.float32([[1,0,100],[0,1,50]])
mouseTransform = cv2.warpAffine(mouseone,M,(cols,rows))
mouseScale = cv2.resize(mouseone,None,fx=0.5, fy=0.5, interpolation = cv2.INTER_CUBIC);
M = cv2.getRotationMatrix2D((cols/2,rows/2),90,1)
mouseRotate = cv2.warpAffine(mouseone,M,(cols,rows))

grayTransform = cv2.cvtColor(mouseTransform, cv2.COLOR_BGR2GRAY)
grayRotate = cv2.cvtColor(mouseRotate, cv2.COLOR_BGR2GRAY)
grayScale = cv2.cvtColor(mouseScale, cv2.COLOR_BGR2GRAY)
momentsTransform = cv2.moments(grayTransform)
huMomentsTransform = cv2.HuMoments(momentsTransform)
momentsRotate = cv2.moments(grayRotate)
huMomentsRotate = cv2.HuMoments(momentsRotate)
momentsScale = cv2.moments(grayScale)
huMomentsScale = cv2.HuMoments(momentsScale)


momentsOne = cv2.moments(grayOne)
huMomentsOne = cv2.HuMoments(momentsOne)

grayTwo = cv2.cvtColor(mousetwo, cv2.COLOR_BGR2GRAY)
momentsTwo = cv2.moments(grayTwo)
huMomentsTwo = cv2.HuMoments(momentsTwo)


cv2.imshow('mouse one',mouseone)


cv2.imshow('mouse two',mousetwo)
#

cv2.imshow('rotate',mouseRotate)
#

cv2.imshow('scale',mouseScale)

cv2.imshow('transform',mouseTransform)

print 'ORIGINAL'
print '----'
print 'M1: '+str(round(momentsOne['m01'],3))+"    M2: "+str(round(momentsOne['m02'],3))+"    M3: "+str(round(momentsOne['m03'],3))
print 'MU1: '+str(round(huMomentsOne[0],12))+"    MU2: "+str(round(huMomentsOne[1],12))+"    MU3: "+str(round(huMomentsOne[2],12))
print 'ALTERNATIVE'
print 'M1: '+str(round(momentsTwo['m01'],3))+"    M2: "+str(round(momentsTwo['m02'],3))+"    M3: "+str(round(momentsTwo['m03'],3))
print 'MU1: '+str(round(huMomentsTwo[0],12))+"    MU2: "+str(round(huMomentsTwo[1],12))+"    MU3: "+str(round(huMomentsTwo[2],12))
print '----'
print 'SCALED'
print 'M1: '+str(round(momentsScale['m01'],3))+"    M2: "+str(round(momentsScale['m02'],3))+"    M3: "+str(round(momentsScale['m03'],3))
print 'MU1: '+str(round(huMomentsScale[0],12))+"    MU2: "+str(round(huMomentsScale[1],12))+"    MU3: "+str(round(huMomentsScale[2],12))
print '----'
print 'ROTATE'
print 'M1: '+str(round(momentsRotate['m01'],3))+"    M2: "+str(round(momentsRotate['m02'],3))+"    M3: "+str(round(momentsRotate['m03'],3))
print 'MU1: '+str(round(huMomentsRotate[0],12))+"    MU2: "+str(round(huMomentsRotate[1],12))+"    MU3: "+str(round(huMomentsRotate[2],12))
print '----'
print 'TRANSFORMED'
print 'M1: '+str(round(momentsTransform['m01'],3))+"    M2: "+str(round(momentsTransform['m02'],3))+"    M3: "+str(round(momentsTransform['m03'],3))
print 'MU1: '+str(round(huMomentsTransform[0],12))+"    MU2: "+str(round(huMomentsTransform[1],12))+"    MU3: "+str(round(huMomentsTransform[2],12))
print '----'

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