# Prenith-and-Drew-Camera-Calibration-Assingment
CSE5283 Warm up assingment to plot the location of a camera based on pictures of a checkerboard calibration image on a sheet of paper.

# Camera Calibration Tool

A Gradio-based interface for camera calibration using chessboard patterns.

## Features
- **Image Processing**: Upload multiple JPEG images containing chessboard patterns
- **Camera Calibration**: Automatically detect corners and compute camera parameters
- **Coordinate Axes Visualization**: Display 3D coordinate frames on calibrated images
- **Undistortion Preview**: Show before/after lens distortion correction
- **Camera Pose Visualization**: Plot camera positions and orientations

## Requirements
```
opencv-python
numpy
gradio
matplotlib
pytransform3d
pillow
```

## Usage
1. **Upload Images**: Select multiple JPEG files containing 9x6 chessboard patterns
2. **Process**: Click "Process Images" to detect corners and calibrate
3. **View Results**: 
   - **Camera Poses**: 3D visualization of camera positions
   - **Sample Images**: Original images with coordinate axes overlaid
   - **Undistortion**: Comparison of original vs corrected images

## File Structure
```
├── Calibration-with-Gradio.ipynb  # Main application
├── utils.py                       # Helper classes (Img, Cam, IO)
├── content/images/               # Processed images directory
└── calibration_results.json     # Saved calibration parameters
```

## Output
- **Camera Matrix**: Intrinsic parameters (focal length, principal point)
- **Distortion Coefficients**: Lens distortion correction parameters
- **Rotation/Translation Vectors**: Individual camera poses for each image
- **JSON Export**: All parameters saved for later use

## Notes
- Chessboard must be 9x6 pattern (9 corners × 6 corners)
- Images should be JPEG format
- Ensure good lighting and multiple viewing angles for best results
