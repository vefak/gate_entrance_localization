import cv2
import numpy as np
import pyzbar.pyzbar as pyzbar
import matplotlib.pyplot as plt

from estimation import homography_est, rigid_body_transform_est


def homogenous_to_img(p):
    p = p / p[2]
    return p[:2]


if __name__ == '__main__':
    # Set up the camera parameters
    K = np.load("./mtx.npy")
    dist = np.load("./dist.npy")
    # Open an open cv camera capture
    capture = cv2.VideoCapture('outpy.m4v')

    # Check if camera opened successfully
    if not capture.isOpened():
        print("Error opening video stream or file")

    # Create the qr code points
    dim = 0.1635 / 2.
    qr_code_points = np.array([[1., -1.], [1., 1.], [-1., 1.], [-1., -1.]]) * dim

    # Get the length of the video sequence
    length = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

    # Create array to store the localization results
    localization = np.zeros((length, 3))
    # Read video until all frames are consumed
    for i in range(length):
        # Get the current frame
        ret, frame = capture.read()

        # Check if there is a current frame
        if ret:
            # Get the QR codes from the image
            codes = pyzbar.decode(frame)

            # If we have detections, pick the first one
            if len(codes) > 0:
                # Extract and draw the polygon points from of the QR codes
                pts = np.array(codes[0].polygon, np.int32)

                # Point ordering sanity check
                ul, bl, br, ur = np.squeeze(pts)[0], np.squeeze(pts)[1], np.squeeze(pts)[2], np.squeeze(pts)[3]
                assert ul[1] < bl[1]
                assert bl[0] < br[0]
                assert ur[1] < br[1]
                assert ul[0] < ur[0]

                # Draw the lines around the QR polygon
                cv2.polylines(frame, [pts], True, (208, 224, 64), 2)

                # Compute the homography that maps the QR code plane into the image plane
                H = homography_est(qr_code_points, pts)

                # Compute the x and y axis of the QR code
                qr_origin = homogenous_to_img(H.dot(np.array([0., 0., 1.])))
                qr_x = homogenous_to_img(H.dot(np.array([0.1, 0., 1.])))
                qr_y = homogenous_to_img(H.dot(np.array([0., 0.1, 1.])))

                # Visualize the x and y axis of the QR code
                cv2.arrowedLine(frame, (int(qr_origin[0]), int(qr_origin[1])), (int(qr_x[0]), int(qr_x[1])),
                                (0, 0, 255), 2, tipLength=0.15)
                cv2.arrowedLine(frame, (int(qr_origin[0]), int(qr_origin[1])), (int(qr_y[0]), int(qr_y[1])),
                                (0, 255, 0), 2, tipLength=0.15)

                # Normalize the coordinates and compute the homography between the camera before the projection and the
                # checkerboard pattern
                pts_homogenous = np.hstack((pts, np.ones((4, 1))))
                pts_homogenous_normalized = (np.linalg.inv(K).dot(pts_homogenous.transpose())).transpose()
                H_normalized = homography_est(qr_code_points, pts_homogenous_normalized[:, :2])
                R, t = rigid_body_transform_est(H_normalized)
                print(t[2])
                print('Translation vector:', -R.transpose() @ t)

                # Store the localization results
                localization[i, :] = -R.transpose() @ t
                x_dist = t[0] # get x[m] value
                z_dist = t[2] # get y[m] value
                print(x_dist,z_dist)

                

            # Display the resulting frame
            cv2.imshow('Frame', frame)

            # Render the frames every second
            cv2.waitKey(100)

        # Break the loop
        else:
            break

    # When everything done, release the video capture object
    capture.release()

    # Close all the open cv frames
    cv2.destroyAllWindows()

    # Choose only the points where z is different than zero
    localization = localization[localization[:, 2] > 0]
    
    # Plot the 2D localization results
    x = localization[:, 0]
    z = localization[:, 2]
 
    plt.figure()
    plt.plot(x, z, 'o')
    plt.grid()
    plt.xlabel('x [m]')
    plt.ylabel('z [m]')
    plt.xlim([-5, 5])
    plt.ylim([-5, 5])
    plt.show()

    plt.figure()
    plt.plot(range(localization.shape[0]), z)
    plt.grid()
    plt.xlabel('Sample')
    plt.ylabel('z [m]')
    plt.ylim([0, 5])
    plt.show()
