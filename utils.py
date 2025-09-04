import os
import json
import glob
import cv2 as cv
import numpy as np

class Img:
    """Camera math: poses, transforms, calibration"""
    
    @staticmethod
    # Takes an array of openCV images
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
            # Find the chess board corners
            ret, corners = cv.findChessboardCorners(img, (board_size[0],board_size[1]), None)

            # If found, add object points, image points (after refining them)
            if ret == True:
                objpoints.append(objp)

                corners2 = cv.cornerSubPix(img,corners, (11,11), (-1,-1), criteria)
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

        return dst

class Cam:
    """Image utilities: loading, preprocessing, format conversion"""
        
    @staticmethod
    def get_coord_frame(rvec, tau, lam, dist) -> np.ndarray:
        W = 2 * np.array([
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
            ], dtype=np.float64)
        
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
    def save_calibration_results(results):
        """Save calibration results to JSON file"""
        path = './calibration.json'

        if (os.path.exists(path)):
            # Convert numpy arrays to lists for JSON serialization
            json_results = {
                'out_points': opoints,
                'in_points': ipoints
            }
            
            with open(path, 'w') as json_file:
                json.dump(json_results, json_file, indent=2)
        else:
            print(f"Error saving calibration results: {e}")
        
        return
    
    @staticmethod
    def load_calibration_results(path):
        """Load calibration results from JSON file"""
        path = './calibration'
        if (os.path.exists(path)):
            data = json.loads(json_str)
            return data
        else:
            print(f"Error loading calibration results: {e}")
            return

    @staticmethod
    def store_images(dir_path):
        # Reads all images and saves them to content/images/
        for file in dir_path:
            image = os.path.basename(file)
            if (image.lower().endswith('.jpeg')) and (os.path.exists(file)):
                # Images read as grayscale
                img = cv.imread(file, cv.IMREAD_GRAYSCALE)

                cv.imwrite('./content/images/' + image, img)

            else:
                print(f'File: {image} must be a .jpeg file.')

        return
            
    @staticmethod
    def get_images():
        # Reads all image files in content/images
        img_list = glob.glob('./content/images/*.jpeg')
        images = []
        for img in img_list:
            img = cv.imread(img, cv.IMREAD_GRAYSCALE)
            images.append(img)
        return images

    @staticmethod
    def remove_images(files):
        # Deletes specified images in
        for frame in files:
            try:
                os.remove('./content/images/' + frame)
            except:
                raise ValueError(f'Could not delete File: {frame}')
        return

