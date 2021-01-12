import glob

import cv2
import numpy as np


def calibrate():
    # Prepare object target points (0, 0, 0), (1, 0, 0), (2, 0, 0), ..., (8, 5, 0) multiplied by the
    # checker dimension in meters
    objp = np.zeros((6 * 9, 3), np.float32)
    objp[:, :2] = np.mgrid[0:6, 0:9].T.reshape(-1, 2) * 0.095

    # Create arrays to store object points and image points from all the images.
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane

    # Set the termination criteria for the corner detection
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # Detect corners in the input images
    images = glob.glob('/home/makman4/Desktop/gate_entrance_localization/data/*.pgm')
    detected = 0

    for fname in images:
        print('Processing frame:', fname)

        # Read input image, resize it to 1080p and convert it to a grayscale image
        img = cv2.resize(cv2.imread(fname), (1920, 1080), interpolation=cv2.INTER_CUBIC)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (6, 9), None)

        # If found, add object points, image points (after refining them)
        if ret == True:
            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)
            detected += 1

    print('Detected:', detected)
    print('Total:', len(images))

    # Calibrate the camera
    img_size = (img.shape[1], img.shape[0])
    ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, cameraMatrix=None, distCoeffs=None)
    print('\nCamera calibrated!')

    # Compute the reprojection error
    total_error = 0.
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], K, dist)
        total_error += cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
    print('Average reprojection error:', total_error / len(objpoints))

    return K, dist


if __name__ == '__main__':
    # Calibrate the camera
    K, dist = calibrate()

    # Print out the obtained calibration parameters
    np.set_printoptions(suppress=True)
    print('K =', K)
    np.save('stereo/mtx.npy', K)
    print('dist =', dist)
    np.save('stereo/dist.npy', dist)
