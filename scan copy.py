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



X = []
y = []

for path in IMAGE_PATH_LIST:
    # cv2.imshow(IMAGE_PATH_LIST)
    print(f"Current path: {path}, Parent name: {path.parent.parent.name}")
    # print(f"processing: {path}")
    features = preprocess_and_extract_features(path)
    if features is not None:
        X.append(features)
        y.append(classes.index(path.parent.parent.name))

# print(X)

X = np.array(X)
y = np.array(y)

# print(y) 

# print(X)


print(f"Number of samples in X: {len(X)}")
print(f"Number of labels in y: {len(y)}")



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training SVM...")
svm_model = SVC(kernel='linear', C=1.0, random_state=42)
svm_model.fit(X_train, y_train)

y_pred = svm_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Classification Report
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=classes))




def test_single_image(image_path, svm_model, classes):
    
    # Preprocess and extract features
    print(f"Testing on image: {image_path}")
    features = preprocess_and_extract_features(image_path)
    
    if features is None:
        print("No features extracted. Unable to classify.")
        return

    features = np.reshape(features, (1, -1))
    
    prediction = svm_model.predict(features)
    predicted_class = classes[prediction[0]]
    
    print(f"Predicted Class: {predicted_class}")

test_image_path = "./Input/acne.jpg"
test_single_image(test_image_path, svm_model, classes)

