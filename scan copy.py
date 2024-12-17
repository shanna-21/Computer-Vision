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



def isskin(image):
    def nothing(x):
        pass

    # Load the image and convert to grayscale
    # gray = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    # mean, std_dev = cv2.meanStdDev(gray)

    # Create a window with trackbars for parameter adjustment
    cv2.namedWindow('image')

    cv2.createTrackbar('HueMin', 'image', 0, 255, nothing)
    cv2.createTrackbar('HueMax', 'image', 20, 255, nothing)
    cv2.createTrackbar('SatMin', 'image', 48, 255, nothing)
    cv2.createTrackbar('SatMax', 'image', 255, 255, nothing)
    cv2.createTrackbar('ValMin', 'image', 80, 255, nothing)
    cv2.createTrackbar('ValMax', 'image', 255, 255, nothing)

    while True:
        # Get current positions of trackbars
        h_min = cv2.getTrackbarPos('HueMin', 'image')
        h_max = cv2.getTrackbarPos('HueMax', 'image')
        s_min = cv2.getTrackbarPos('SatMin', 'image')
        s_max = cv2.getTrackbarPos('SatMax', 'image')
        v_min = cv2.getTrackbarPos('ValMin', 'image')
        v_max = cv2.getTrackbarPos('ValMax', 'image')

        # Convert the image to HSV color space
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Define lower and upper bounds for skin detection
        lower_hsv = np.array([h_min, s_min, v_min])
        upper_hsv = np.array([h_max, s_max, v_max])

        # Create a binary mask where skin regions are white
        skinMask = cv2.inRange(hsv_image, lower_hsv, upper_hsv)

        # Optional: Apply morphological operations to clean up the mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        skinMask = cv2.erode(skinMask, kernel, iterations=2)
        skinMask = cv2.dilate(skinMask, kernel, iterations=2)

        # Blur the mask to smooth the edges
        skinMask = cv2.GaussianBlur(skinMask, (5, 5), 0)

        # Apply the mask to the original image
        skin = cv2.bitwise_and(image, image, mask=skinMask)

        alpha = np.uint8(skinMask > 0) * 255
        result = cv2.merge((skin, alpha))  # Combine BGR with alpha

        # Save the resulting image
        cv2.imwrite("skin_detection_result.png", result)
        
        # Return the processed image
        return result

def preprocess_and_extract_features(image_path):

    img = cv2.imread(str(image_path))

    if img is None:
        print(f"Failed to load {image_path}. Skipping.")
        return None, None
    
    final = isskin(img)
    img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, threshold1=100, threshold2=200)
    final = cv2.equalizeHist(edges)
    

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

svm_y_pred = svm_model.predict(X_test)
svm_accuracy = accuracy_score(y_test, svm_y_pred)
print(f"Accuracy: {svm_accuracy * 100:.2f}%")

# Classification Report
print("Classification Report:")
print(classification_report(y_test, svm_y_pred, target_names=classes))

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

test_image_path = "./Input/eyebags1.jpg"
test_single_image(test_image_path, svm_model, classes)

print("**" * 20)


print("Training Random Forest...")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

rf_y_pred = rf_model.predict(X_test)
rf_accuracy = accuracy_score(y_test, rf_y_pred)
print(f"Accuracy: {rf_accuracy * 100:.2f}%")

# Classification Report
print("Classification Report:")
print(classification_report(y_test, rf_y_pred, target_names=classes))

test_single_image(test_image_path, rf_model, classes)

print("**" * 20)

print("Training XGB...")
xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
xgb_model.fit(X_train, y_train)

xgb_y_pred = xgb_model.predict(X_test)
xgb_accuracy = accuracy_score(y_test, xgb_y_pred)
print(f"Accuracy: {xgb_accuracy * 100:.2f}%")

# Classification Report
print("Classification Report:")
print(classification_report(y_test, xgb_y_pred, target_names=classes))

test_single_image(test_image_path, xgb_model, classes)

print("**" * 20)

base_models = [
    ('svm', svm_model),
    ('rf', rf_model),
    ('xgb', xgb_model)
]

meta_model1 = LogisticRegression(random_state=42)

stacking_model1 = StackingClassifier(estimators=base_models, final_estimator=meta_model1)

print("Stacking Model with Logistic Regression")
stacking_model1.fit(X_train, y_train)

stacking_y_pred = stacking_model1.predict(X_test)
stacking_accuracy = accuracy_score(y_test, stacking_y_pred)
print(f"Accuracy: {stacking_accuracy * 100:.2f}%")

print("Classification Report:")
print(classification_report(y_test, stacking_y_pred, target_names=classes))

test_single_image(test_image_path, stacking_model1, classes)

print("**" * 20)

meta_model2 = RandomForestClassifier(random_state=42)
stacking_model2 = StackingClassifier(estimators=base_models, final_estimator=meta_model2)

print("Stacking Model with Random Forest Classifier")
stacking_model2.fit(X_train, y_train)

stacking_y_pred = stacking_model2.predict(X_test)
stacking_accuracy = accuracy_score(y_test, stacking_y_pred)
print(f"Accuracy: {stacking_accuracy * 100:.2f}%")

print("Classification Report:")
print(classification_report(y_test, stacking_y_pred, target_names=classes))

test_single_image(test_image_path, stacking_model2, classes)

print("**" * 20)



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
