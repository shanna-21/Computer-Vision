from pathlib import Path
import numpy as np
import os
import cv2
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from pathlib import Path
from skimage.feature import hog, local_binary_pattern, graycomatrix, graycoprops
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
    h_min = 0
    h_max = 128
    s_min = 100
    s_max = 150
    v_min = 0
    v_max = 128

    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    lower_hsv = np.array([h_min, s_min, v_min])
    upper_hsv = np.array([h_max, s_max, v_max])
    
    skinMask = cv2.inRange(hsv_image, lower_hsv, upper_hsv)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    skinMask = cv2.erode(skinMask, kernel, iterations=2)
    skinMask = cv2.dilate(skinMask, kernel, iterations=2)

    skinMask = cv2.GaussianBlur(skinMask, (5, 5), 0)

    skin = cv2.bitwise_and(image, image, mask=skinMask)

    alpha = np.uint8(skinMask > 0) * 255
    result = cv2.merge((skin, alpha))

    return result

def preprocess_and_extract_features(image_path, IMG_SIZE=128):

    img = cv2.imread(str(image_path))
    img = isskin(img)

    if img is None:
        print(f"Failed to load {image_path}. Skipping.")
        return None

    img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, threshold1=100, threshold2=200)
    final = cv2.equalizeHist(edges)

    # HOG Feature Extraction
    def extract_hog_features(img, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2)):
        features, _ = hog(
            img,
            orientations=orientations,
            pixels_per_cell=pixels_per_cell,
            cells_per_block=cells_per_block,
            block_norm='L2-Hys',
            visualize=True
        )
        return features

    # LBP Feature Extraction
    def extract_lbp_features(img, P=8, R=1):
        lbp = local_binary_pattern(img, P=P, R=R, method='uniform')
        hist, _ = np.histogram(lbp, bins=np.arange(0, P + 3), range=(0, P + 2))
        hist = hist.astype('float')
        hist /= (hist.sum() + 1e-6)
        return hist

    # Color Histogram Feature Extraction
    def extract_color_features(img, h_range=(0, 128), s_range=(100, 150), v_range=(0, 128)):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hist_h = cv2.calcHist([hsv], [0], None, [256], h_range)
        hist_s = cv2.calcHist([hsv], [1], None, [256], s_range)
        hist_v = cv2.calcHist([hsv], [2], None, [256], v_range)
        hist_h /= hist_h.sum() + 1e-6
        hist_s /= hist_s.sum() + 1e-6
        hist_v /= hist_v.sum() + 1e-6
        return np.concatenate([hist_h.flatten(), hist_s.flatten(), hist_v.flatten()])

    # GLCM Feature Extraction
    def extract_glcm_features(img, distances=[5], angles=[0], properties=('contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation')):
        glcm = graycomatrix(img, distances=distances, angles=angles, levels=256, symmetric=True, normed=True)
        glcm_features = []
        for prop in properties:
            glcm_features.append(graycoprops(glcm, prop).flatten())
        return np.concatenate(glcm_features)

    # Combined Feature Extraction
    def extract_combined_features(img):
        lbp_features = extract_lbp_features(gray)
        hog_features = extract_hog_features(final)
        color_features = extract_color_features(img_resized)
        glcm_features = extract_glcm_features(gray)
        return np.concatenate([lbp_features, hog_features, color_features, glcm_features])

    combined_features = extract_combined_features(img_resized)

    return combined_features


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

test_image_path = "./Input/eyebags2.jpg"
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
        ('svm', SVC(kernel='linear', probability=True)),
        ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
        ('xgb', XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42))
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

meta_model3 = SVC(random_state=42)
stacking_model3 = StackingClassifier(estimators=base_models, final_estimator=meta_model3)

print("Stacking Model with SVC")
stacking_model3.fit(X_train, y_train)

stacking_y_pred = stacking_model3.predict(X_test)
stacking_accuracy = accuracy_score(y_test, stacking_y_pred)
print(f"Accuracy: {stacking_accuracy * 100:.2f}%")

print("Classification Report:")
print(classification_report(y_test, stacking_y_pred, target_names=classes))

test_single_image(test_image_path, stacking_model3, classes)

print("" * 20)

meta_model4 = XGBClassifier(random_state=42)
stacking_model4 = StackingClassifier(estimators=base_models, final_estimator=meta_model4)

print("Stacking Model with XGB")
stacking_model4.fit(X_train, y_train)

stacking_y_pred = stacking_model4.predict(X_test)
stacking_accuracy = accuracy_score(y_test, stacking_y_pred)
print(f"Accuracy: {stacking_accuracy * 100:.2f}%")

print("Classification Report:")
print(classification_report(y_test, stacking_y_pred, target_names=classes))

test_single_image(test_image_path, stacking_model4, classes)

print("" * 20)