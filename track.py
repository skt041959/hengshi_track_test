import cv2
import numpy as np
import IPython
import sys

a1 = (1, (786, 303, 11, 11))
a2 = (2, (787, 291, 11, 14))
a3 = (3, (788, 279, 11, 14))
a4 = (4, (788, 268, 12, 11))

comm = sys.argv[1]
s = int(sys.argv[2])

img1 = cv2.imread('{0}/{0}{1:04}.bmp'.format(comm, s), cv2.CV_LOAD_IMAGE_GRAYSCALE)
img2 = cv2.imread('{0}/{0}{1:04}.bmp'.format(comm, s+1), cv2.CV_LOAD_IMAGE_GRAYSCALE)

diff = cv2.absdiff(img1, img2)
diff2 = cv2.GaussianBlur(diff, (3,3), 0)
retval,bin1 = cv2.threshold(diff2, 5, 255, cv2.THRESH_BINARY)

k = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))

bin2 = cv2.dilate(bin1, k)
bin2 = cv2.erode(bin2, k)
bin2 = cv2.dilate(bin2, k)
print(bin2.nonzero()[0].size)

w = bin2.nonzero()[1].max()-bin2.nonzero()[1].min()
h = bin2.nonzero()[0].max()-bin2.nonzero()[0].min()
#y1 = bin2.nonzero()[0].min()-w
#y2 = bin2.nonzero()[0].max()+w
#x1 = bin2.nonzero()[1].min()-h
#x2 = bin2.nonzero()[1].max()+h
y1 = bin2.nonzero()[0].min()
y2 = bin2.nonzero()[0].max()
x1 = bin2.nonzero()[1].min()
x2 = bin2.nonzero()[1].max()
print(w, h, y1, y2, x1, x2)
roi = img2[y1:y2, x1:x2]

eage = cv2.Canny(roi, 1000, 4000, apertureSize=5)
cv2.namedWindow('w', cv2.WINDOW_NORMAL)
cv2.imshow('w', eage)
cv2.waitKey(10)

contours0, hier0 = cv2.findContours(eage, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

mask = np.zeros((1004, 992), np.uint8)
roi_mask = mask[y1:y2, x1:x2]

cv2.drawContours(roi_mask, contours0, -1, (255), -1)
print(mask.nonzero()[0].size)
#mask = cv2.erode(mask, None)
#print(mask.nonzero()[0].size)

hist = cv2.calcHist([img2], [0], mask, [32], [0, 256])

cv2.normalize(hist, hist, 0, 255, cv2.NORM_MINMAX)
hist.reshape(-1)

track_window = (mask.nonzero()[1].min(), 
        mask.nonzero()[0].min(),
        mask.nonzero()[1].max()-mask.nonzero()[1].min(),
        mask.nonzero()[0].max()-mask.nonzero()[0].min())

term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 1 )
cv2.namedWindow('track', cv2.WINDOW_NORMAL)
cv2.namedWindow('origin', cv2.WINDOW_NORMAL)

for i in range(s+1, 200):
    frame = cv2.imread('{1}/{1}{0:04}.bmp'.format(i, comm))
    back = cv2.calcBackProject([frame], [0], hist, [0,256], 1)
    track_box, track_window = cv2.CamShift(back, track_window, term_crit)
    print(i, track_window, track_box)
    #cv2.ellipse(back, track_box, (255), 2)
    cv2.ellipse(frame, track_box, (0, 0, 255), 2)
    cv2.imshow('track', back)
    cv2.imshow('origin', frame)
    key = cv2.waitKey(10)
    if key == ord('q'):
        break
    if key == ord('s'):
        cv2.imwrite('track{0:04}.bmp'.format(i), frame)
    #IPython.embed()

cv2.destroyWindow('track')
cv2.destroyWindow('origin')

