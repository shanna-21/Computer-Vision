import cv2
import numpy as np

# Load the image
image = cv2.imread('./Input/red girl.jpg')

# Load pre-trained face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Convert to grayscale for face detection
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect faces
faces = face_cascade.detectMultiScale(image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

if len(faces) == 0:
    print('no face')

for (x, y, w, h) in faces:
    # Extract the face region
    face = image[y:y+h, x:x+w]

    # Convert face region to HSV color space
    hsv = cv2.cvtColor(face, cv2.COLOR_BGR2HSV)

    # Define red color range in HSV
    lower_red1 = np.array([0, 50, 50])   # Lower bound for red
    upper_red1 = np.array([10, 255, 255])  # Upper bound for red
    lower_red2 = np.array([170, 50, 50])  # Red can wrap around HSV range
    upper_red2 = np.array([180, 255, 255])

    # Create masks for red color
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = cv2.add(mask1, mask2)

    # Calculate the percentage of red pixels
    red_pixels = cv2.countNonZero(red_mask)
    total_pixels = face.shape[0] * face.shape[1]
    redness_percentage = (red_pixels / total_pixels) * 100

    print(f"Redness percentage in the detected face: {redness_percentage:.2f}%")

    # Display the mask
    cv2.imshow('Original Face', face)
    cv2.imshow('Red Mask', red_mask)

# Show results and wait for key press
cv2.waitKey(0)
# cv2.destroyAllWindows()





# from pathlib import Path
# import numpy as np
# import os
# import cv2
# from skimage.feature import hog
# from tqdm import tqdm

# IMAGE_PATH = Path("./Dataset")
# print(f"Checking path: {IMAGE_PATH.resolve()}")  

# if not IMAGE_PATH.exists():
#     print(f"The directory '{IMAGE_PATH}' does not exist.")
# else:
#     IMAGE_PATH_LIST = list(IMAGE_PATH.glob("**/*.jpg"))
#     print(f'Total Images = {len(IMAGE_PATH_LIST)}')

# classes = os.listdir(IMAGE_PATH)
# classes = sorted(classes)

# print(classes)

# print("**" * 20)
# print(" " * 10, f"Total Classes = {len(classes)}")
# print("**" * 20)

# for c in classes:
#     total_images_class = list(Path(os.path.join(IMAGE_PATH, c)).glob("*/*.jpg"))
#     print(f"* {c}: {len(total_images_class)} images")

# IMG_SIZE = 128

# def preprocess_and_extract_features(image_path):
#     img = cv2.imread(str(image_path))

#     if img is None:
#         print(f"Failed to load {image_path}. Skipping.")
#         return None, None
    
#     img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    
#     gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)

#     Z = img_resized.reshape((-1, 3))  # Flatten image
#     Z = np.float32(Z)
    
#     # Define criteria and apply kmeans()
#     criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
#     k = 4  # Number of clusters (adjust based on your needs)
#     _, labels, centers = cv2.kmeans(Z, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    
#     centers = np.uint8(centers)
#     segmented_img = centers[labels.flatten()]
#     segmented_img = segmented_img.reshape(img_resized.shape)  # Reshape back to the image shape

#     # Convert the segmented image to grayscale
#     gray_segmented = cv2.cvtColor(segmented_img, cv2.COLOR_BGR2GRAY)

#     final = cv2.equalizeHist(gray_segmented)

#     sift = cv2.SIFT_create()
#     keypoints, descriptors = sift.detectAndCompute(final, None)
    
#     if descriptors is None:
#         print(f"No SIFT features detected in {image_path}")
#         return None
    
#     return descriptors


# # Store descriptors for all images
# descriptors_list = []
# image_paths = []
# labels = []

# for path in IMAGE_PATH_LIST:
#     print(f"Processing {path}")
#     descriptors = preprocess_and_extract_features(path)
#     if descriptors is not None:
#         descriptors_list.append(descriptors)
#         image_paths.append(path)
#         labels.append(path.parent.name)

# # Convert descriptors list to a single numpy array
# all_descriptors = np.vstack(descriptors_list)

# # FLANN parameters
# index_params = dict(algorithm=1, trees=10)  # Use KD-tree algorithm
# search_params = dict(checks=50)  # Number of checks during search

# # FLANN-based matcher
# flann = cv2.FlannBasedMatcher(index_params, search_params)

# # Add all descriptors to FLANN index
# flann.add([all_descriptors])
# flann.train()

# def classify_image_with_flann(image_path, flann, image_paths, labels):
#     print(f"Testing on image: {image_path}")
    
#     # Extract features from the query image
#     query_descriptors = preprocess_and_extract_features(image_path)
#     if query_descriptors is None:
#         print("No descriptors extracted. Unable to classify.")
#         return

#     # Find the nearest neighbor matches between query descriptors and dataset descriptors
#     matches = flann.knnMatch(query_descriptors, all_descriptors, k=2)

#     # Apply ratio test as in Lowe's SIFT paper
#     good_matches = []
#     for m, n in matches:
#         if m.distance < 0.9 * n.distance:
#             good_matches.append(m)

#     # print(good_matches)
#     if len(good_matches) < 5:  # If not enough good matches, return unknown
#         print("Not enough good matches found.")
#         return "Unknown"

#     # Find the most common label for the best matches
#     match_labels = []
#     for img_idx, match in enumerate(good_matches):
#         # img_idx = match.trainIdx
#         match_labels.append(labels[img_idx])

#     # Return the most frequent class
#     predicted_class = max(set(match_labels), key=match_labels.count)
#     print(f"Predicted Class: {predicted_class}")
#     return predicted_class


# test_image_path = "./Input/red girl.jpg"
# predicted_class = classify_image_with_flann(test_image_path, flann, image_paths, labels)

