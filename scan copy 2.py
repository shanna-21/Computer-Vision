from pathlib import Path
import numpy as np
import os
import cv2
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from pathlib import Path
from skimage.feature import hog
from skimage.feature import local_binary_pattern
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder

IMAGE_PATH = Path("./Dataset")
print(f"Checking path: {IMAGE_PATH.resolve()}")  

if not IMAGE_PATH.exists():
    print(f"The directory '{IMAGE_PATH}' does not exist.")
else:
    IMAGE_PATH_LIST = list(IMAGE_PATH.glob("**/*.jpg"))
    print(f'Total Images = {len(IMAGE_PATH_LIST)}')

# print(IMAGE_PATH_LIST)

classes = os.listdir(IMAGE_PATH)
classes = sorted(classes)

print(classes)

print("**" * 20)
print(" " * 10, f"Total Classes = {len(classes)}")
print("**" * 20)

for c in classes:
    total_images_class = list(Path(os.path.join(IMAGE_PATH, c)).glob("*/*.jpg"))
    print(f"* {c}: {len(total_images_class)} images")


IMG_SIZE = 128
def preprocess_and_extract_features(image_path):

    img = cv2.imread(str(image_path))

    if img is None:
        print(f"Failed to load {image_path}. Skipping.")
        return None, None
    
    img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, threshold1=100, threshold2=200)
    final = cv2.equalizeHist(edges)
    # edges = cv2.Canny(gray, threshold1=100, threshold2=200)


    # ---------------- SIFT ---------------
    # sift = cv2.SIFT_create()
    # keypoints, descriptors = sift.detectAndCompute(gray, None)
    
    # if descriptors is None:
    #     print(f"No SIFT features detected in {image_path}")
    #     return None
    
    # feature_vector = np.mean(descriptors, axis=0)
    # return feature_vector


    # ---------------- SURF ---------------
    # surf = cv2.xfeatures2d.SURF_create(hessianThreshold=400)

    # # Detect keypoints and compute descriptors
    # keypoints, descriptors = surf.detectAndCompute(gray, None)
    
    # if descriptors is None:
    #     print(f"No SURF features detected in {image_path}")
    #     return None

    # # Optionally, use the mean of descriptors as a feature vector
    # feature_vector = np.mean(descriptors, axis=0)

    # return feature_vector
    # >>>GABISA<<<



    # ---------------- HOG ---------------
    hog_features, hog_image = hog(
        final,
        orientations=9,  # Number of orientation bins
        pixels_per_cell=(8, 8),  # Size of the cell
        cells_per_block=(2, 2),  # Number of cells per block
        block_norm='L2-Hys',  # Normalization method
        visualize=True,  # Return HOG image for debugging (optional)
        transform_sqrt=True  # Apply power law compression
    )

    return hog_features



    # ---------------- LBP ---------------
    # radius = 3  # Radius of the circular neighborhood
    # n_points = 8 * radius  # Number of points in the neighborhood

    # # Extract LBP features
    # lbp = local_binary_pattern(final, n_points, radius, method="uniform")
    
    # # Calculate the histogram of LBP
    # n_bins = int(lbp.max() + 1)
    # hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins))

    # # Normalize the histogram
    # hist = hist.astype("float")
    # hist /= (hist.sum() + 1e-7)  # Avoid division by zero

    # return hist

def create_cnn_model(input_shape, num_classes):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# Reshape HOG features for CNN
def reshape_features_for_cnn(features):
    num_samples = features.shape[0]
    feature_size = features.shape[1]
    size = int(np.sqrt(feature_size))
    return features.reshape((num_samples, size, size, 1))  # Reshape for CNN input



def visualize_features(image_path, svm_model, classes):
    img = cv2.imread(image_path)
    
    if img is None:
        print(f"Failed to load {image_path}. Skipping.")
        return
    
    features = preprocess_and_extract_features(image_path)
    
    features = np.reshape(features, (1, -1))
    prediction = xgb_model.predict(features)
    predicted_class = classes[prediction[0]]
    
    print(f"Predicted Class: {predicted_class}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # REDNESS
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_red = np.array([0, 50, 50])
    upper_red = np.array([10, 255, 255])
    mask = cv2.inRange(hsv, lower_red, upper_red)
    # redness = cv2.bitwise_and(img, img, mask=mask)
    # cv2.imshow('red', redness)

    # ACNE
    acne_area = cv2.inRange(gray, 80, 255)  # Adjust based on your image's acne color/texture
    cv2.imshow('acne', acne_area)

    # BAGS
    hsv_bags = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_eyebags = np.array([0, 0, 100])
    upper_eyebags = np.array([180, 50, 200])
    mask_eyebags = cv2.inRange(hsv, lower_eyebags, upper_eyebags)
    # eyebags = cv2.bitwise_and(img, img, mask_eyebags=mask_eyebags)

    if predicted_class == 'Bags':
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2) 
            cv2.putText(img, "Face", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        contours, _ = cv2.findContours(mask_eyebags, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            cv2.drawContours(img, [contour], -1, (0, 0, 255), 2)
    
        print('Bags')
    elif predicted_class == 'Redness':
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(img, "Face", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            cv2.drawContours(img, [contour], -1, (0, 0, 255), 2)
    elif predicted_class == 'Acne':
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(img, "Face", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        contours, _ = cv2.findContours(acne_area, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            cv2.drawContours(img, [contour], -1, (255, 0, 0), 2)

    cv2.imshow("Annotated Image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Example usage
test_image_path = "./Input/red girl 3.png"
visualize_features(test_image_path, svm_model, classes)
