# Panorama Stitcher

# About
This project lets you stitch an arbitrary amount of pictures together into a panorama

# Dependencies
This project depends on **multiprocess** (_NOT_ multiprocessing), openCV 4+ and numpy.

To install both, simply run:
```
pip install multiprocess
pip install opencv-python
pip install numpy
```

## Usage Guide:

To use this repo, first clone the repo using:
```
git clone https://github.com/Ryan-Red/PanoramaStitcher.git
```

**Important**: The first image **MUST** be the top-left image in the desired panorama. Please bear in mind that the default resolution of the resulting panorama is defined in the top of the `mosaic_images.py' file under the variables `MAX_RES_X` and `MAX_RES_Y`. This will be changed in a future version. 

## Example:
We have 3 images: 1.png, 2.png and 3.png inside a folder called img.

1.png is the left-most picture in the set, so to generate the panorama, we'd call (assuming you are outside of src):
```
python3 src/panorama.py img/1.png img/2.png img/3.png
```
or 
```
python3 src/panorama.py img/*
```


# Enjoy!
