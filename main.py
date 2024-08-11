import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import io
import trimesh
import mediapipe as mp
import math

# Loading or creating a 3D mesh for the glases.
f = 'glasses.obj'  # Ensure this file is in working directory
mesh = trimesh.load(f, process=False)  # Load the 3D mesh
scene = mesh.scene()  # Create a scene from the mesh

# Render the 3D mesh to a PNG image
data = scene.save_image()  # Save the rendered image data
eyeimage = np.array(Image.open(io.BytesIO(data)))  # Convert image data to a numpy array

# Process the rendered image to extract the glasses region
kernel = np.ones((50, 300), np.uint8)  # Create a kernel for processing
s_h, s_w, _ = eyeimage.shape  # Get the dimensions of the image
gray = cv2.cvtColor(eyeimage, cv2.COLOR_BGR2GRAY)  # Convert to grayscale

# Find the vertical boundaries of the glasses region.
startingPt = -1
endingPt = -1
x = 150  # Central x-coordinate for region extraction.

for y in range(50, s_h - 50, 1):
    array = gray[y - 25:y + 25, x - 100:x + 200] * kernel  # Extract a region
    average = np.mean(array)  # Calculate the average intensity.

    if average != 255:
        if startingPt == -1:
            startingPt = y
        else:
            endingPt = y

cropped_image = eyeimage[startingPt:endingPt, :]  # Croping the glasses image
s_h, s_w, _ = cropped_image.shape  # Update dimensions after cropping

# Initialize MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# Open webcam for live face detection
cap = cv2.VideoCapture(0)  # Starting video capture from webcam.

while True:
    # Capture a frame from the webcam
    ret, faceImg = cap.read()
    if not ret:
        break  # Exit loop if no frame is captured

    imageHeight, imageWidth, _ = faceImg.shape  # Get the dimensions of the capured frame.

    # Process the face image for face landmarks
    with mp_face_detection.FaceDetection(min_detection_confidence=0.5) as face_detection:
        results = face_detection.process(cv2.cvtColor(faceImg, cv2.COLOR_BGR2RGB))  # Convert to RGB and detect faces
        if results.detections:
            for detection in results.detections:
                # Extract face landmarks
                nose_tip = mp_face_detection.get_key_point(detection, mp_face_detection.FaceKeyPoint.NOSE_TIP)
                left_ear = mp_face_detection.get_key_point(detection, mp_face_detection.FaceKeyPoint.LEFT_EAR_TRAGION)
                right_ear = mp_face_detection.get_key_point(detection, mp_face_detection.FaceKeyPoint.RIGHT_EAR_TRAGION)
                left_eye = mp_face_detection.get_key_point(detection, mp_face_detection.FaceKeyPoint.LEFT_EYE)
                right_eye = mp_face_detection.get_key_point(detection, mp_face_detection.FaceKeyPoint.RIGHT_EYE)

                # Convert normalized coordinates to pixel coordinates
                Nose_tip_x, Nose_tip_y = mp_drawing._normalized_to_pixel_coordinates(nose_tip.x, nose_tip.y, imageWidth, imageHeight)
                Left_Ear_x, Left_Ear_y = mp_drawing._normalized_to_pixel_coordinates(left_ear.x, left_ear.y, imageWidth, imageHeight)
                Right_Ear_x, Right_Ear_y = mp_drawing._normalized_to_pixel_coordinates(right_ear.x, right_ear.y, imageWidth, imageHeight)
                Left_EYE_x, Left_EYE_y = mp_drawing._normalized_to_pixel_coordinates(left_eye.x, left_eye.y, imageWidth, imageHeight)
                Right_EYE_x, Right_EYE_y = mp_drawing._normalized_to_pixel_coordinates(right_eye.x, right_eye.y, imageWidth, imageHeight)

                # Calculate the rotation angle for the sunglasses
                dx = Right_EYE_x - Left_EYE_x
                dy = Right_EYE_y - Left_EYE_y
                angle = 180 - math.degrees(math.atan2(dy, dx))  # Angle in degrees

                # Calculate sunglasses dimensions and resize
                sunglass_width = Left_Ear_x - Right_Ear_x
                sunglass_height = int((s_h / s_w) * sunglass_width)
                imgFront = cv2.resize(cropped_image, (sunglass_width, sunglass_height), None, 0.3, 0.3)  # Resize with scaling factor

                # Rotate the sunglasses image to align with the face
                center = (imgFront.shape[1] // 2, imgFront.shape[0] // 2)
                rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
                rotated_imgFront = cv2.warpAffine(imgFront, rotation_matrix, (imgFront.shape[1], imgFront.shape[0]), flags=cv2.INTER_LINEAR)

                # Prepare for overlaying sunglasses on the face
                y_adjust = int((sunglass_height / 80) * 100)  # Fine-tune vertical adjustment
                x_adjust = int((sunglass_width / 194) * 100)  # Fine-tune horizontal adjustment
                pos = [Nose_tip_x - x_adjust, Nose_tip_y - y_adjust]

                hf, wf, cf = rotated_imgFront.shape
                hb, wb, cb = faceImg.shape

                # Create mask for sunglasses overlay
                *_, mask = cv2.split(rotated_imgFront)
                maskBGRA = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGRA)
                maskBGR = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
                imgRGBA = cv2.bitwise_and(rotated_imgFront, maskBGRA)
                imgRGB = cv2.cvtColor(imgRGBA, cv2.COLOR_BGRA2BGR)

                imgMaskFull = np.zeros((hb, wb, cb), np.uint8)
                imgMaskFull[pos[1]:hf + pos[1], pos[0]:wf + pos[0], :] = imgRGB
                imgMaskFull2 = np.ones((hb, wb, cb), np.uint8) * 255
                maskBGRInv = cv2.bitwise_not(maskBGR)
                imgMaskFull2[pos[1]:hf + pos[1], pos[0]:wf + pos[0], :] = maskBGRInv

                # Blend the sunglasses with the face image
                faceImg2 = faceImg.copy()
                for y in range(0, hb):
                    for x in range(0, wb):
                        if imgMaskFull[y, x, 0] != 0 and imgMaskFull[y, x, 0] != 255:
                            faceImg2[y, x, :] = 0.3 * faceImg2[y, x, :] + 0.7 * imgMaskFull[y, x, :]

                # Display the result
                cv2.imshow('Sunglasses Overlay', faceImg2)

    # Exit loop when 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release webcam and close all windows
cap.release()
cv2.destroyAllWindows()
