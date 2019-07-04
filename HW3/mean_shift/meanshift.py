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

# https://www.bogotobogo.com/python/OpenCV_Python/python_opencv3_mean_shift_tracking_segmentation.php
# https://docs.opencv.org/2.4/modules/imgproc/doc/histograms.html?highlight=calcbackproject
# https://www.learnopencv.com/color-spaces-in-opencv-cpp-python/

def tracking_mean_shift(path_to_dataset, roi_range, hue_range = (0, 180), debug = False):
    frame = imread(path_to_dataset + '/0001.jpg')
    (x,y,w,h) = roi_range
    track_window = roi_range

    # set up the ROI for tracking
    roi = frame[y:y+h, x:x+w]

    # convert BGR image to HSV
    hsv_roi =  cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # set threshold the HSV image to get certain color
    # H – Hue ( Dominant Wavelength ).
    # S – Saturation ( Purity / shades of the color ).
    # V – Value ( Intensity ).
    start_hue = hue_range[0]
    end_hue = hue_range[1]
    mask = cv2.inRange(hsv_roi, np.array((start_hue, 60.,32.)), np.array((end_hue,255.,255.)))
    
    # images : it is the source image of type uint8 or float32. it should be given in square brackets, ie, "[img]".
    # channels : it is also given in square brackets. It is the index of channel for which we calculate histogram. 
    #            For example, if input is grayscale image, its value is [0]. For color image, you can pass [0], [1] or [2] 
    #            to calculate histogram of blue, green or red channel respectively.
    # mask : mask image. To find histogram of full image, it is given as "None". But if you want to find histogram 
    #        of particular region of image, you have to create a mask image for that and give it as mask.
    # histSize : this represents our BIN count. Need to be given in square brackets. For full scale, we pass [256].
    # ranges : this is our RANGE. Normally, it is [0,256].
    roi_hist = cv2.calcHist([hsv_roi],[0],mask,[end_hue-start_hue],[start_hue, end_hue])
    
    # src – input array.
    # dst – output array of the same size as src .
    # alpha – norm value to normalize to or the lower range boundary in case of the range normalization.
    # beta – upper range boundary in case of the range normalization; it is not used for the norm normalization.
    # normType – normalization type (see the details below).
    # dtype – when negative, the output array has the same type as src; otherwise, it has the same number of channels as src and the depth =CV_MAT_DEPTH(dtype).
    # mask – optional operation mask.
    cv2.normalize(roi_hist,roi_hist,0,255,cv2.NORM_MINMAX)
    
    if debug == True:
        print("Hue range: ", start_hue, end_hue)
        print("Frame, roi, roi in hsv, mask")
        plt.subplot(321), plt.imshow(frame, 'gray')
        plt.subplot(322), plt.imshow(roi, 'gray')
        plt.subplot(323), plt.imshow(hsv_roi,'gray')
        plt.subplot(324), plt.imshow(mask, 'gray')
        plt.show()
        print("ROI hist")
        plt.plot(roi_hist)
        plt.show()
        print("ROI hist without mask")
        create_and_draw_hist(frame)

    # Setup the termination criteria, either 10 iteration or move by at least 1 pt
    term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 )
    
    draw_first_back = debug

    for filename in sorted(glob.glob(path_to_dataset + '/*.jpg')):
        frame = imread(filename)
        ret = True
        if ret == True:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

# images – Source arrays. They all should have the same depth, CV_8U or CV_32F , and the same size. 
#          Each of them can have an arbitrary number of channels.
# nimages – Number of source images.
# channels – The list of channels used to compute the back projection. The number of channels must match the histogram dimensionality. 
#            The first array channels are numerated from 0 to images[0].channels()-1 , 
#            the second array channels are counted from images[0].channels() to images[0].channels() + images[1].channels()-1, and so on.
# hist – Input histogram that can be dense or sparse.
# backProject – Destination back projection array that is a single-channel array of the same size and depth as images[0] .
# ranges – Array of arrays of the histogram bin boundaries in each dimension. See calcHist() .
# scale – Optional scale factor for the output back projection.
# uniform – Flag indicating whether the histogram is uniform or not (see above).
            prob_image = cv2.calcBackProject([hsv], [0], roi_hist, [start_hue, end_hue], 1)
    
#             if draw_first_back == True :
#                 print("Back ")
#                 plt.plot(), plt.imshow(prob_image, 'gray')
#                 plt.show()
#                 draw_first_back = False

            # apply meanshift to get the new location
            ret, track_window = cv2.meanShift(prob_image, track_window, term_crit)

            # Draw it on image
            x,y,w,h = track_window
            img2 = cv2.rectangle(frame, (x,y), (x+w,y+h), 255,2)
            cv2.imshow('img2', img2)

            k = cv2.waitKey(60) & 0xff
            if k == 27:
                break
            else:
                cv2.imwrite(chr(k)+".jpg",img2)

        else:
            break

    cv2.destroyAllWindows()
    cv2.waitKey(1)

def main():
    path = sys.argv[1]
    roi = tuple(map(int, sys.argv[2].split(',')))
    hue_range = (0,180)
    if (len(sys.argv) == 4):
    	hue_range = tuple(map(int, sys.argv[3].split(',')))
	
    print(path)
    print(type(roi), roi, type(roi[0]), roi[0])
    print(hue_range)
    tracking_mean_shift(path, roi, hue_range)

main()