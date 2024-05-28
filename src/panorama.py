# Add other imports from available listing as your see fit.
from ctypes.wintypes import HACCEL
import sys
import numpy as np
from cv2 import imread, imwrite, resize, imshow, waitKey, destroyAllWindows

from mosaic_images import mosaic_images


# PANORAMA_WIDTH = 3000
# PANORAMA_HEIGHT = 1300
# PANORAMA_CHANNELS = 3

def create_panorama(image_files):
    
    # Load images.
    images = []
    for i in range(0, len(image_files), 1):
        i_name = image_files[i] 
        images.append(imread(i_name))

    
    Ipano = images[0]
    
    for idx in range(1, len(images)):

        # Undo vignetting on input 
        Ipano = mosaic_images(Ipano, images[idx])

        

    imshow("result",Ipano)
    waitKey()
    destroyAllWindows()

    # imwrite(Ipano)

if __name__ == "__main__":
    # Make sure the right number of input args are provided.
    # if len(sys.argv) != 3:
    #     sys.exit(-1, "Too few command line argmuments.")

    create_panorama(sys.argv[1:])