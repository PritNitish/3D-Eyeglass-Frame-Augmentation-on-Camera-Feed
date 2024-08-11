**3D Eyeglass Frame Augmentation on CameraFeed**

**Overview**

               Thisproject demonstrates how to overlay a 3D glasses model on a live webcam feedusing     Python, OpenCV, and MediaPipe.The code captures a video stream from the webcam, detects              facial landmarks, calculates therequired rotation and position for the glasses, and overlays the       glasses onto the detected face.

**Prerequisites**

1\. Create a Conda Environment

·       Use the following command tocreate a new environment:

·       conda create --name mainpython=3.11

2\. Install Libraries

·       Use the attached YAML file toinstall the required libraries, or you can manually install them using pip:

·       pip install numpy opencv-pythonmatplotlib Pillow trimesh mediapipe

 **Files**

·       glasses.obj: 3D model file forthe glasses (ensure this file is in your working directory).

·       main.py: Python script for theproject.

**How It Works**

1\. Load and Render 3D Glasses Model

·       The glasses.obj file is loaded,and a 3D scene is created.

·       The scene is rendered to a PNGimage, which is then processed to extract the glasses region.

2\. Face Detection and Glasses Overlay

·       The webcam feed is captured.

·       Face landmarks are detectedusing MediaPipe.

·       The rotation angle and positionfor the glasses are calculated based on the eye landmarks.

·       The glasses image is resized,rotated, and overlaid onto the face in the webcam feed.

3\. Display

·       The result with the glassesoverlay is displayed in a window.

·       Press 'q' to exit the windowand stop the video capture.

**Running the Script**

1\. Ensure that you have the \`glasses.obj\`file in your working directory.

2\. Run the script:

  python main.py

3\. The webcam feed will open, and theglasses will be overlaid on detected faces.

**Code Explanation**

·       Loading 3D Model: Uses trimeshto load and render the 3D glasses model.

·       Processing Rendered Image:Converts the rendered image to grayscale and extracts the glasses region.

·       Face Detection: UtilizesMediaPipe to detect face landmarks.

·       Calculating Rotation andPosition: Determines the angle and position to overlay the glasses based on eyelandmarks.

·       Overlaying Glasses: Adjusts theglasses image and blends it with the face image in the webcam feed.

**Troubleshooting**

·       Ensure that your webcam isworking and accessible.

·       Verify that \`glasses.obj\` iscorrectly placed in your working directory.

·       If the glasses do not aligncorrectly, you may need to adjust the resizing and rotation parameters.
## Output




https://github.com/user-attachments/assets/5eb29fad-4356-44c3-8e0e-593de74845a5

