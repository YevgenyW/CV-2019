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

# Object tracking with camshift
```python meanshift.py path_to_dataset roi hue_range```
# where 'hue_range' - again optional parameter.

In some cases when meanshift provides good results, camshift has worse results. But in general camshift adapts windows size.

Meanshift can lose object when object and background has the same color parameters. It's well seen on the example with 'Bird1' dataset. It's necessary to control histogram parameters in order to provide better results in this case (like I did with custom third parameter). But, such "custom control" doesn't work in general cases.
