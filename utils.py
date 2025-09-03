import os
import json
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

class Img:
    """Camera math: poses, transforms, calibration"""
    
    @staticmethod
    # Takes a string of openCV images
    def find_chessboard_corners(images, board_size=[9, 6]):
        """Returns list of points in real space and on the image plane"""
        # termination criteria
        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        objp = np.zeros((board_size[1]*board_size[0],3), np.float32)
        objp[:,:2] = np.mgrid[0:board_size[0],0:board_size[1]].T.reshape(-1,2)

        # Arrays to store object points and image points from all the images.
        objpoints = [] # 3d point in real world space
        imgpoints = [] # 2d points in image plane.

        for img in images:
            gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

            # Find the chess board corners
            ret, corners = cv.findChessboardCorners(gray, (board_size[0],board_size[1]), None)

            # If found, add object points, image points (after refining them)
            if ret == True:
                objpoints.append(objp)

                corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
                imgpoints.append(corners2)

        return objpoints, imgpoints
    
    @staticmethod
    # Takes openCV Image
    def undistort_image(image, lam, dist):
        h, w = image.shape[:2]
        newcameramtx, roi = cv.getOptimalNewCameraMatrix(lam, dist, (w,h), 1, (w,h))

        # undistort
        dst = cv.undistort(image, lam, dist, None, newcameramtx)

        # crop the image
        x, y, w, h = roi
        dst = dst[y:y+h, x:x+w]

        return dist

class Cam:
    """Image utilities: loading, preprocessing, format conversion"""
        
    @staticmethod
    def get_coord_frame(omega, tau, lam, dist) -> np.ndarray:
        W = 2 * np.array([
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
            ], dtype=np.float64)
        
        rvec = cv.Rodrigues(omega)[0]
        
        image_axes, _ = cv.projectPoints(W, rvec, tau, lam, dist)
        image_axes = image_axes.squeeze().T

        return image_axes

    @staticmethod
    def get_camera_pos (image, omega, tau) -> np.ndarray:
        pose = np.block([omega.T, -omega.T @ tau])
        pose = np.vstack([pose, [0, 0, 0, 1]])

        return pose

class IO:
    """File and device I/O utilities"""
    
    @staticmethod
    def save_calibration_results(results: dict, path: str) -> bool:
        """Save calibration results to JSON file"""
        try:
            # Convert numpy arrays to lists for JSON serialization
            json_results = {
                'reprojection_error': float(results['ret']),
                'camera_matrix': results['mtx'].tolist(),
                'distortion_coefficients': results['dist'].tolist(),
                'rotation_vectors': [r.tolist() for r in results['rvecs']],
                'translation_vectors': [t.tolist() for t in results['tvecs']]
            }
            
            with open(path, 'w') as f:
                json.dump(json_results, f, indent=2)
            return True
        except Exception as e:
            print(f"Error saving calibration results: {e}")
            return False
    
    @staticmethod
    def load_calibration_results(path: str) -> Optional[dict]:
        """Load calibration results from JSON file"""
        try:
            with open(path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading calibration results: {e}")
            return None

    @staticmethod
    def get_images():
        return

    @staticmethod
    def upload_images():
        return

    @staticmethod
    def clear_img_dir()
