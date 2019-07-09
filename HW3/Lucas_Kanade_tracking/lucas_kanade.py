import sys
import numpy as np
import cv2
from matplotlib import pyplot as plt
import glob

def gray(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

def imread(imgpath):
    return cv2.cvtColor(cv2.imread(imgpath), cv2.COLOR_BGR2RGB)

def imshow(img, cmap=None):
    plt.title(img.shape)
    plt.imshow(img, cmap=cmap)
    plt.axis('off')
    
def create_and_draw_hist(img_RGB):
    img_HSV =  cv2.cvtColor(img_RGB, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([img_HSV],[0],None,[255],[0, 255])
#     cv2.normalize(hist,hist,0,255,cv2.NORM_MINMAX)
    plt.plot(hist)
    plt.show()

def tracking_lucas_canade(path_to_dataset, roi_range, pyrLevel = 0):
    old_gray = imread(path_to_dataset + '/0001.jpg')

    x = roi_range[0]; y = roi_range[1]; w = roi_range[2]; h = roi_range[3]
    
    # Lucas kanade params
    lk_params = dict(winSize = (w, h), 
                     maxLevel = pyrLevel, 
                     criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    old_points = np.array([[x, y]], dtype=np.float32)

    for filename in sorted(glob.glob(path_to_dataset + '/*.jpg')):
        gray_frame = imread(filename)
        
        new_points, status, error = cv2.calcOpticalFlowPyrLK(old_gray, gray_frame, old_points, None, **lk_params)
        old_gray = gray_frame.copy()
        
        old_points = new_points

        frame = cv2.imread(filename)

        x1, y1 = new_points[0].ravel()

        cv2.rectangle(frame, (x1,y1), (int(x1)+w,int(y1)+h), 255,2)

        cv2.imshow("Frame", frame)
        
        key = cv2.waitKey(1)
        if key == 27:
            break

    cv2.destroyAllWindows()
    cv2.waitKey(1)

def main():
    path = sys.argv[1]
    roi = tuple(map(int, sys.argv[2].split(',')))
    pyrLevel = 0
    if (len(sys.argv) == 4):
    	pyrLevel = int(sys.argv[3])
	
    print(path)
    print(roi)
    print(pyrLevel)
    tracking_lucas_canade(path, roi, pyrLevel)

main()