# Script to start object tracking with Lucas-Kanade algorithm in general form
```python meanshift.py path_to_dataset roi pyrLevel```

# Example:
```python lucas_kanade.py '../data/Biker/img/' 262,94,16,26 5```
# where ../data/Biker/img/' - path to folder with images;
# 198,214,34,81 - roi;
# 5 - pyramid level number(by default 0). 0 means pyramids are not used (single level). Optional parameter.

A couple of words about algorithm implementation. It's typical prectise to get roi, find features inside roi (with cv.goodFeaturesToTrack for example) and then created optical flow based on these features. But in my implementation I used just one point to track (left upper corner of roi) and create roi based on this point. This approach show better results on the datasets.

Lucas-Kanade algorithm works well on BlurBody, Biker and Surfer datasets.
It loses tracking object on Bird1 and BlurCar2 dataset because the algorithm isn't able to get matrix W correctly due to fast-changing background.

In general Lucas-Kanade works better than meanshift.