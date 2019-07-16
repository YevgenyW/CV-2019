# Script to start object tracking with meanshift in general form
```python meanshift.py path_to_dataset roi hue_range```

# Example:
```python meanshift.py '../data/Basketball/img/' 198,214,34,81 50,70```
# where '../data/Basketball/img/' - path to folder with images;
# '198,214,34,81' - roi;
# '50,70' - hue range (by default (0, 180)).

Third parameter is used just for better results. I found the following optimal value of this parameter for the following datasets:
'data/Basketball/img/' - (50, 70)
'data/Biker/img/' - (90, 120)
'data/BlurBody/img/' - (10, 30)
'data/BlurCar2/img/' - (125,130)
'data/Bird1/img/' - (92, 97)
'data/Surfer/img/' - (0, 255)

I added two movies (Bird1 and Basketball) to show how algorithm works with custom parameters.

# Object tracking with camshift
```python meanshift.py path_to_dataset roi hue_range```
# where 'hue_range' - again optional parameter.

In general, meanshift can lose object if roi and background has similar color histogram or roi changes its position too quickly.

'Basketball', 'Bird1' and 'BlurBody' datasets are the typical examples of the first reason.
'Biker' and 'BlurBody' datasets - examples of the second reason.

'Surfer' dataset has good result because roi object is contrast and doesn't change its location too quickly.

First reason could be fixed by custom parameters of histogram parameters (like I did with custom third parameter). But, such "custom control" doesn't work for general cases.
