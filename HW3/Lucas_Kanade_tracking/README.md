# Script to start object tracking with Lucas-Kanade algorithm in general form
```python meanshift.py path_to_dataset roi pyrLevel```

# Example:
```python lucas_kanade.py '../data/Biker/img/' 262,94,16,26 5```
# where ../data/Biker/img/' - path to folder with images;
# 198,214,34,81 - roi;
# 5 - pyramid level number(by default 0). 0 means pyramids are not used (single level). Optional parameter.