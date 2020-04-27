import numpy as np
import cv2
import matplotlib.image as mpimg
import glob

# Read in and make a list of calibration images
images = glob.glob('camera_cal/calibration*.jpg')

# Array to store object points and image points from all the images

objpoints = []  # 3D points in real world space
imgpoints = []  # 2D points in image plane

cols = 9
rows = 6

def calib():
    """
    To get an undistorted image, we need camera matrix & distortion coefficient
    Calculate them with 9*6 20 chessboard images
    """
    # Prepare object points
    objp = np.zeros((rows * cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)  # x,y coordinates

    for fname in images:

        img = cv2.imread(fname)
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (cols, rows), None)

        # If corners are found, add object points, image points
        if ret == True:
            imgpoints.append(corners)
            objpoints.append(objp)

            # 체스보드의 결과값을 확인하고 싶다면 'python -O calibration.py'를 실행한다.
            if __debug__==False:
                # Draw and display the corners
                img = cv2.drawChessboardCorners(img, (cols, rows), corners, ret)
                cv2.imshow('drawChessBoard',img)
                cv2.waitKey(0)
        else:
            continue

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    if ret<0:
        print('Error : cv2.calibrateCamera')

    return mtx, dist

def undistort(img, mtx, dist):
    """ undistort image """
    return cv2.undistort(img, mtx, dist, None, mtx)


mtx, dist = calib()

for i, fname in enumerate(images):
    img = cv2.imread(fname)

    cv2.imshow('img', img)
    # Use the undistort function, with the mtx and dist calculated above
    undist = undistort(img, mtx, dist)
    cv2.imshow('undist', undist)

    cv2.waitKey()
cv2.destroyAllWindows()