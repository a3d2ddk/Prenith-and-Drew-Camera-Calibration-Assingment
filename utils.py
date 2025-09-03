import cv2 as cv
import numpy as np
import json
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Tuple, Optional
import gradio as gr
import os

class Cam:
    """Camera math: poses, transforms, calibration"""
    
    @staticmethod
    def calibrate_camera(obj_points: List, img_points: List, img_shape: Tuple) -> Tuple:
        """Camera calibration with proper return"""
        ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(
            obj_points, img_points, img_shape, None, None
        )
        return ret, mtx, dist, rvecs, tvecs
    
    @staticmethod
    def prepare_chessboard_points(board_size: Tuple[int, int], square_size: float) -> np.ndarray:
        """Generate 3D chessboard object points"""
        objp = np.zeros((board_size[0] * board_size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:board_size[0], 0:board_size[1]].T.reshape(-1, 2)
        objp *= square_size
        return objp
    
    @staticmethod
    def find_chessboard_corners(img: np.ndarray, board_size: Tuple[int, int]) -> Tuple[bool, Optional[np.ndarray]]:
        """Find and refine chessboard corners"""
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        ret, corners = cv.findChessboardCorners(gray, board_size, None)
        
        if ret:
            criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        
        return ret, corners
    
    @staticmethod
    def undistort_image(img: np.ndarray, mtx: np.ndarray, dist: np.ndarray) -> np.ndarray:
        """Undistort image using camera matrix and distortion coefficients"""
        return 

class Img:
    """Image utilities: loading, preprocessing, format conversion"""
    
    @staticmethod
    def load_image(path: str) -> Optional[np.ndarray]:
        """Load image from path"""
        return cv.imread(path)
    
    @staticmethod
    def bgr_to_rgb(img: np.ndarray) -> np.ndarray:
        """Convert BGR to RGB for display"""
        return cv.cvtColor(img, cv.COLOR_BGR2RGB)
    
    @staticmethod
    def save_image(img: np.ndarray, path: str) -> bool:
        """Save image to path"""
        return cv.imwrite(path, img)
    
    @staticmethod
    def get_image_shape(img: np.ndarray) -> Tuple[int, int]:
        """Get image dimensions (width, height)"""
        h, w = img.shape[:2]
        return (w, h)
    
    @staticmethod
    def resize_image(img: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
        """Resize image to specified size"""
        return cv.resize(img, size)

class Render:
    """Rendering utilities: drawing overlays, coordinate systems"""
    
    @staticmethod
    def draw_chessboard_corners(img: np.ndarray, board_size: Tuple[int, int], 
                               corners: np.ndarray) -> np.ndarray:
        """Draw chessboard corners on image"""
        img_copy = img.copy()
        cv.drawChessboardCorners(img_copy, board_size, corners, True)
        return img_copy
    
    @staticmethod
    def project_points_to_image(points_3d: np.ndarray, rvec: np.ndarray, 
                               tvec: np.ndarray, mtx: np.ndarray, 
                               dist: np.ndarray) -> np.ndarray:
        """Project 3D points to 2D image coordinates"""
        img_points, _ = cv.projectPoints(points_3d, rvec, tvec, mtx, dist)
        return img_points.reshape(-1, 2).astype(int)
    
    @staticmethod
    def project_coordinate_axes(rvec: np.ndarray, tvec: np.ndarray,
                               mtx: np.ndarray, dist: np.ndarray,
                               axis_length: float = 50.0) -> np.ndarray:
        """Project 3D coordinate axes to 2D image points"""
        axes_points = np.float32([
            [0, 0, 0],              # Origin
            [axis_length, 0, 0],    # X-axis
            [0, axis_length, 0],    # Y-axis
            [0, 0, -axis_length]    # Z-axis
        ]).reshape(-1, 3)
        
        return Render.project_points_to_image(axes_points, rvec, tvec, mtx, dist)
    
    @staticmethod
    def draw_coordinate_frame(image_points, img):

        """ Draw coordinate frame on image"""
        x0, y0 = image_points[:,0].astype(int)
        cv.circle(img, (x0, y0), 9, (0, 0, 0), -1)

        x1, y1 = image_points[:,1].astype(int)
        img = cv.arrowedLine(img, (x0, y0), (x1, y1), (255, 0, 0), 5)

        x2, y2 = image_points[:,2].astype(int)
        img = cv.arrowedLine(img, (x0, y0), (x2, y2), (0, 255, 0), 5)

        x3, y3 = image_points[:,3].astype(int)
        img = cv.arrowedLine(img, (x0, y0), (x3, y3), (0, 0, 255), 5)

    @staticmethod
    def draw_coordinate_axes(img: np.ndarray, rvec: np.ndarray, tvec: np.ndarray,
                           mtx: np.ndarray, dist: np.ndarray, 
                           axis_length: float = 50.0) -> np.ndarray:
        """Draw 3D coordinate axes on image (convenience function)"""
        image_points = Render.project_coordinate_axes(rvec, tvec, mtx, dist, axis_length)
        return Render.draw_coordinate_frame(image_points, img)
    
    @staticmethod
    def create_camera_pose_plot(rvecs: List, tvecs: List) -> plt.Figure:
        """Create 2D plot of camera positions"""
        fig = rvecs
        return fig

class IO:
    """File and device I/O utilities"""
    
    @staticmethod
    def ensure_directory(path: str) -> None:
        """Create directory if it doesn't exist"""
        os.makedirs(path, exist_ok=True)
    
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
    def get_file_paths(directory: str, extension: str = ".jpg") -> List[str]:
        """Get all files with given extension in directory"""
        path = Path(directory)
        return [str(p) for p in path.glob(f"*{extension}")]
