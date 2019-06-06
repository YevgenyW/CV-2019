import cv2
import numpy as np

def gaussian_2d_func(x, y, sigma):
    return 1.0/(2.0*np.pi*sigma**2) * np.exp((x**2+y**2)/(-2.0*sigma**2))

def gaussian_2d_kernel(rx, ry, sigma):
    kernel = np.zeros([ry*2+1, rx*2+1])

    for y in range(-ry, ry + 1):
        for x in range(-rx, rx + 1):
            kernel[x+rx,y+ry] = gaussian_2d_func(x, y, sigma)
            
    return kernel

def gaussian_blur(img, kernel_sz, sigma):
    res = np.asarray(img)
    width = img.shape[1]
    height = img.shape[0]
    ry = int(kernel_sz[0]/2)
    rx = int(kernel_sz[1]/2)

    kernel = gaussian_2d_kernel(ry, rx, sigma)
    img = np.pad(img, (ry, rx), 'constant', constant_values=(0, 0))
    res = np.zeros_like(img)
    
    for y in range(ry, height + ry):
        for x in range(rx, width+rx):
            res[y, x] = (kernel * img[y-ry : y+ry+1, x-rx : x+rx+1]).sum()
    
    return res[ry:-ry, rx:-rx]

def is_keypoint_fast(img, x, y):
    rel_coords = [(0, -3), (3, 0), (0, 3), (-3, 0)]
    
    threshold = 10
    max_threshold = img[x][y] + threshold
    min_threshold = img[x][y] - threshold
    
    num_darker = 0
    num_lighter = 0
    
    for point in rel_coords:
        value = img[x + point[0]][y + point[1]]
        if (value > max_threshold):
            num_darker+=1
        elif (value < min_threshold):
            num_lighter+=1
    
    return (num_darker >= 3 or num_lighter >= 3)
    
def is_keypoint(img, x, y):
    rel_coords = [(0, -3),(1, -3), (2, -2), (3, -1), (3, 0), (3, 1), (2, 2), (1, 3), 
                  (0, 3), (-1, 3), (-2, 2), (-3, 1), (-3, 0), (-3, -1), (-2, -2), (-1, -3)]
    
    num_darker = 0
    num_lighter = 0
    
    threshold = 10
    max_threshold = img[x][y] + threshold
    min_threshold = img[x][y] - threshold
    lighter_seq = 0
    darker_seq = 0
    
    num_points_limit = 9
    
    for point in rel_coords:
        value = img[x + point[0]][y + point[1]]
        if (value > max_threshold):
            darker_seq = 0
            lighter_seq+=1
            if (lighter_seq >= num_points_limit):
                return True
            
        elif (value < min_threshold):
            darker_seq+=1
            lighter_seq = 0
            if (darker_seq >= num_points_limit):
                return True
        
    return False
        
def keypoints_detect(img):
#     https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_feature2d/py_fast/py_fast.html
    key_points = []
    
    width = img.shape[1]
    height = img.shape[0]
    start_shift = 3
    
    for y in range(start_shift, height - start_shift):
        for x in range(start_shift, width - start_shift):
            if (is_keypoint_fast(img, x, y) == True):
                if (is_keypoint(img, x, y) == True):
                    key_points.append((y, x))
    
    return key_points
    
def find_keypoints_candidates(img):
#     img_blur = gaussian_blur(img, (31, 31), 3.0)
    return keypoints_detect(img)

sigma = 5
num_points = 256
gaussian_x_a = np.random.normal(loc=0.0, scale=sigma, size=num_points).astype(np.int)
gaussian_y_a = np.random.normal(loc=0.0, scale=sigma, size=num_points).astype(np.int)
gaussian_x_b = np.random.normal(loc=0.0, scale=sigma, size=num_points).astype(np.int)
gaussian_y_b = np.random.normal(loc=0.0, scale=sigma, size=num_points).astype(np.int)

def compute_descriptors(img, kp_arr):
    width = img.shape[1]
    height = img.shape[0]
    descrs = np.zeros((len(kp_arr), 32)).astype('uint8')

    kp_idx = 0
    
    for kp in kp_arr:
    # some tips about algorithm from https://github.com/skanti/BRIEF-Binary-Descriptor-AVX2
        kp_x = kp[1]
        kp_y = kp[0]
        
        if ((kp_x <= sigma*3) or (kp_x >= (width - sigma*3))
               or (kp_y <= sigma*3) or (kp_y >= (height - sigma*3))):
            continue
           
        cos_angle = np.cos(kp_idx*5/180.0*np.pi)
        sin_angle = np.sin(kp_idx*5/180.0*np.pi)
        
        ia_x = np.zeros(num_points)
        ia_y = np.zeros(num_points)
        ib_x = np.zeros(num_points)
        ib_y = np.zeros(num_points)
        
        for i in range(num_points):
            ia_x[i] = int(gaussian_x_a[i]*cos_angle - gaussian_y_a[i]*sin_angle)
            ia_y[i] = int(gaussian_x_a[i]*sin_angle + gaussian_y_a[i]*cos_angle)
            ib_x[i] = int(gaussian_x_b[i]*cos_angle - gaussian_y_b[i]*sin_angle)
            ib_y[i] = int(gaussian_x_b[i]*sin_angle + gaussian_y_b[i]*cos_angle)
        
        for i in range(32):
            value = 0
            for j in range(8):
                idx_a_y = int(kp_y + ia_y[i*8 + j])
                idx_a_x = int(kp_x + ia_x[i*8 + j])
                idx_b_y = int(kp_y + ib_y[i*8 + j])
                idx_b_x = int(kp_x + ib_x[i*8 + j])
                bit = (img[idx_a_y][idx_a_x] < img[idx_b_y][idx_b_x]) << j
                value|=bit 
            
            descrs[kp_idx][i] = value
        
        kp_idx+=1
        
    return descrs

def match_hamming(descr_arr0, descr_arr1):
    from scipy.spatial import distance
    arr1 = descr_arr0
    arr2 = descr_arr1
    if (descr_arr1.shape[0] < descr_arr0[0].shape[0]):
        arr1 = descr_arr1
        arr1 = descr_arr0

    matches_arr = []
    p1_idx = 0
    
    for point1 in arr1:
        
        min_idx = p2_idx = 0
        min_distance = distance.hamming(point1, arr2[min_idx])
        
        for point2 in arr2:
            cur_distance = distance.hamming(point1, point2)
            if (cur_distance < min_distance):
                min_idx = p2_idx
                min_distance = cur_distance
            p2_idx+=1
        
        if(min_distance < 0.75):    
            matches_arr.append((p1_idx, min_idx))
        
        p1_idx+=1
                
                
    return matches_arr

# function for keypoints and descriptors calculation
def detect_keypoints_and_calculate_descriptors(img):
    # img - numpy 2d array (grayscale image)
    img_blur = gaussian_blur(img, (31, 31), 3.0)

    # keypoints
    kp_arr = find_keypoints_candidates(img_blur)
    # kp_arr is array of 2d coordinates-tuples, example:
    # [(x0, y0), (x1, y1), ...]
    # xN, yN - integers

    # descriptors
    descr_arr = compute_descriptors(img_blur, kp_arr)
    # cv_descr_arr is array of descriptors (arrays), example:
    # [[v00, v01, v02, ...], [v10, v11, v12, ...], ...]
    # vNM - floats

    return kp_arr, descr_arr
