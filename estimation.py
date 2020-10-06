import numpy as np


def homography_est(src_pts, target_pts):
    # Creat the A matrix
    A = np.zeros((8, 9))
    for i in range(4):
        # Grab the corner coordinates
        x1 = src_pts[i, 0]
        x2 = src_pts[i, 1]
        y1 = target_pts[i, 0]
        y2 = target_pts[i, 1]

        # Populate two rows of the A matrix
        A[2 * i, :] = np.array([-x1, -x2, -1, 0, 0, 0, x1 * y1, x2 * y1, y1])
        A[2 * i + 1, :] = np.array([0, 0, 0, -x1, -x2, -1, x1 * y2, x2 * y2, y2])

    # Compute the SVD
    _, _, Vh = np.linalg.svd(A)
    V = Vh.transpose()

    # Extract h
    h = V[:, 8]

    # Extract H from h
    H = np.reshape(h, (3, 3))

    return H / H[2, 2]


def rigid_body_transform_est(H):
    # Extract h1, h2 and h3 from H
    h1 = H[:, 0]
    h2 = H[:, 1]
    h3 = H[:, 2]

    # Construct R prime
    R_prime = np.hstack((h1.reshape(-1, 1), h2.reshape(-1, 1), (np.cross(h1, h2)).reshape(-1, 1)))

    # Recover R from R prime
    # NOTE: The normalization with det(U * V') ensures just that we get a right
    # handed coordinate system as result (i.e., to enforce that det(R) = 1).
    U, _, Vh = np.linalg.svd(R_prime, full_matrices=False)
    D = np.array([[1., 0., 0], [0., 1., 0.], [0., 0., np.linalg.det(U.dot(Vh))]])
    R = U @ D @ Vh

    # Recover t
    t = h3 / np.linalg.norm(h1)

    return R, t
