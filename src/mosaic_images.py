# Add other imports from available listing as your see fit.
import sys
import numpy as np
from multiprocess import Pool
import time


from numpy.linalg import inv
from cv2 import imread, imwrite, resize, imshow, waitKey, destroyAllWindows
from cv2 import SIFT_create
from cv2 import FlannBasedMatcher
from cv2 import findHomography, RANSAC

# TODO: remove this import

MAX_RES_X = 3000
MAX_RES_Y = 1300


def bilinear_interp(I, pt):
    """
    Performs bilinear interpolation for a given image point.

    Given the (x, y) location of a point in an input image, use the surrounding
    four pixels to conmpute the bilinearly-interpolated output pixel intensity.

    Note that images are (usually) integer-valued functions (in 2D), therefore
    the intensity value you return must be an integer (use round()).

    This function is for a *single* image band only - for RGB images, you will 
    need to call the function once for each colour channel.

    Parameters:
    -----------
    I   - Single-band (greyscale) intensity image, 8-bit np.array (i.e., uint8).
    pt  - 2x1 np.array of point in input image (x, y), with subpixel precision.

    Returns:
    --------
    b  - Interpolated brightness or intensity value (whole number >= 0).
    """
    #--- FILL ME IN ---
   
    if pt.shape != (2, 1):
        raise ValueError('Point size is incorrect.')

    x = pt[0]
    y = pt[1]

    # Get the left-most and right-most pixel coord to use for interp:
    x_1 = (int) (np.floor(x))
    x_2 = (int) (np.ceil(x))

    dx = x_2 - x_1

    # Do the same for heighest and lowest pixels:
    y_1 = (int) (np.floor(y))
    y_2 = (int) (np.ceil(y))

    dy = y_2 - y_1


    f11 = I[y_1,x_1]
    f12 = I[y_2,x_1]
    f21 = I[y_1,x_2]
    f22 = I[y_2,x_2]

    # Interpolation:

    # x-interpolation
    f_x_y_1 = (x_2 - x)/dx * f11 + (x - x_1)/dx * f21
    f_x_y_2 = (x_2 - x)/dx * f12 + (x - x_1)/dx * f22

    # y interpolation

    f_x_y = (y_2 - y)/dy * f_x_y_1 + (y - y_1)/dy * f_x_y_2


    # Round and return the result
    b = (int)(np.round(f_x_y))


    #------------------

    return b


def mosaic_images(I1, I2, show_me = True):
    # Load images.
    # I1 = imread(I1_name)#.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # I2 = imread(I2_name)#.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Feature matching - remember I1 is the anchor.
    feature_descriptor = SIFT_create()
    keypoints1, descriptors1 = feature_descriptor.detectAndCompute(I1, None)
    keypoints2, descriptors2 = feature_descriptor.detectAndCompute(I2, None)

    # if show_me:
        #   imshow("Key points in Image 1", drawKeypoints(I1, keypoints1, None))
        #   imshow("Key points in Image 2", drawKeypoints(I2, keypoints2, None))
        #   waitKey()
        #   destroyAllWindows()

    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 2)
    search_params = dict(checks = 50)
    
    # print(descriptors1.shape, descriptors2.shape)



    flann = FlannBasedMatcher(index_params,search_params)
   
    matches = flann.knnMatch(np.float32(descriptors1), np.float32(descriptors2), k=2)
    # store all the good matches as per Lowe's ratio test.
    good = []
    notFinished = True
    thresh = 0.35
    while(notFinished):
        for m,n in matches:
            if m.distance < thresh*n.distance:
                good.append(m)
        if(len(good) < 50):
            thresh += 0.05
        else:
            break
        # good = good[:50]

    print("Amount of good points:", len(good), "versus total number of matches:", len(matches) )

    src_pts = np.float32([ keypoints1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
    dst_pts = np.float32([ keypoints2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
    M, mask = findHomography(src_pts, dst_pts, RANSAC, 5.0)

    M_inv =  inv(M)

    I3 = np.zeros((MAX_RES_Y, MAX_RES_X, 3), dtype=I1.dtype)

    I3[:I1.shape[0], :I1.shape[1]] = I1



    topLeft =  np.array([0,0,1])
    topLeft = M_inv @ topLeft
    topLeft = np.array([[topLeft[0]/topLeft[2],topLeft[1]/topLeft[2]]]).T

    topRight =  np.array([I2.shape[1],0,1])
    topRight = M_inv @ topRight
    topRight = np.array([[topRight[0]/topRight[2],topRight[1]/topRight[2]]]).T

    bottomLeft =  np.array([0,I2.shape[0],1])
    bottomLeft = M_inv @ bottomLeft
    bottomLeft = np.array([[bottomLeft[0]/bottomLeft[2],bottomLeft[1]/bottomLeft[2]]]).T

    bottomRight =  np.array([I2.shape[1],I2.shape[0],1])
    bottomRight = M_inv @ bottomRight
    bottomRight = np.array([[bottomRight[0]/bottomRight[2],bottomRight[1]/bottomRight[2]]]).T

    min_x = int(np.min([topLeft[0], topRight[0],bottomLeft[0], bottomRight[0] ]))
    min_y = int(np.min([topLeft[1], topRight[1],bottomLeft[1], bottomRight[1] ]))

    max_x = int(np.max([topLeft[0], topRight[0],bottomLeft[0], bottomRight[0] ]))
    max_y = int(np.max([topLeft[1], topRight[1],bottomLeft[1], bottomRight[1] ]))

    min_x = int(np.max([min_x,0]))
    min_y = int(np.max([min_y,0]))


    max_x = int(np.min([max_x, I1.shape[1] + I2.shape[1], MAX_RES_X]))
    max_y = int(np.min([max_y, I1.shape[0] + I2.shape[0], MAX_RES_Y]))

    n_process = 10

    indices = np.arange(min_y, max_y)
    I3[:I1.shape[0], :I1.shape[1]] = I1

    def computeImg(j):
        I3_temp = np.zeros((1, MAX_RES_X,3), dtype='uint8')
        for i in range(min_x,max_x,1): #x index
            
            val = I3[j, i, 0] == I3[j, i, 1] == 0 and I3[j, i, 2] == 0
            if(val == False):
                continue
            
            pointToTransform = np.array([i,j,1])
            target = np.matmul(M, pointToTransform)
            t = np.array([[target[0]/target[2],target[1]/target[2]]]).T

            if(t[0] > 0 and t[0] < I2.shape[1] - 1 and t[1] > 0 and t[1] < I2.shape[0] -1):
                resR = bilinear_interp(I2[:,:,0],t)
                resG = bilinear_interp(I2[:,:,1],t)
                resB = bilinear_interp(I2[:,:,2],t)
                if(resR == resG == resB == 0):
                    continue
                I3_temp[0, i, 0] = resR
                I3_temp[0, i, 1] = resG
                I3_temp[0, i, 2] = resB

        return I3_temp

    p = Pool(n_process)
    result = p.map_async(computeImg, indices)
    
    I3_aggregated = result.get(timeout=60)
    
    print(len(I3_aggregated), max_y)

    I3_added = np.vstack(I3_aggregated)
    I3_integrated = np.zeros((MAX_RES_Y,MAX_RES_X,3),dtype='uint8')
    I3_integrated[min_y:max_y, :, :] = I3_added
    I3 = I3 + I3_integrated

    

            



    # for j in range(0,I2.shape[0],1): # y index, iterates inside the bounding box
    #         for i in range(0,I2.shape[1],1): #x index
                
                
    #             try:
    #                 val = I2[j, i, 0] == I2[j, i, 1] == 0 and I2[j, i, 2] == 0
            
    #                 if (val) :
    #                     continue
    #             except Exception as e:
    #                 continue
    #             pointToTransform = np.array([i,j,1])
    #             target = np.matmul(M_inv, pointToTransform)
    #             t = np.array([[target[0]/target[2],target[1]/target[2]]]).T

              
    #             if t[0] > maxT_x:
    #                 maxT_x = int(t[0])
    #             if t[1] > maxT_y:
    #                 maxT_y = int(t[1])

                

    #             # # if(t[1] < 0):
    #             # print(t)
    #             # resR = bilinear_interp(I1[:,:,0],t)
    #             # resG = bilinear_interp(I1[:,:,1],t)
    #             # resB = bilinear_interp(I1[:,:,2],t)
                
    #             t = t.astype(np.int64)
        
    #             # # print(t)
    #             try:
    #                 I3[t[1],t[0],0] = I2[j,i,0]
    #                 I3[t[1],t[0],1]= I2[j,i,1]
    #                 I3[t[1],t[0],2] = I2[j,i,2]
    #             except Exception as e:
  
    #                 try:
    #                     I3[t[1],t[0],0] = I3[j,i,0]
    #                     I3[t[1],t[0],1] = I3[j,i,1]
    #                     I3[t[1],t[0],2] = I3[j,i,2]
    #                 except Exception as e:
    #                     continue

    #                 continue

       

    # dst = perspectiveTransform(pts,M_inv) ############################################################### NOOOOOOOOO!!!!!

    


    # img3 = warpPerspective(I2, M, (w,h))
    
    


    # I1 = polylines(I1,[np.int32(dst)],True,255,3, LINE_AA)


    # draw_params = dict(matchColor = (0,255,0), # draw matches in green color
    #                 singlePointColor = None,
    #                 matchesMask = matchesMask, # draw only inliers
    #                 flags = 2)
    # img3 = drawMatches(I1,keypoints1,I2,keypoints2,good,None,**draw_params)
    # I3 = I3[:MAX_RES_Y,:MAX_RES_X]
    # imshow('gray', I3)

 
    # waitKey()
    # destroyAllWindows()

    # imshow('gray2', I3_added)
    # waitKey()
    # destroyAllWindows()
    # Get the Homography







    # Return mosaicked images (of correct, full size).
    return I3

if __name__ == "__main__":
    # Add test code here if you desire. 
    show_me = len(sys.argv) > 3
    mosaic_images(sys.argv[1], sys.argv[2], show_me)
    pass