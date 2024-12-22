import streamlit as st
import pickle
import numpy as np
import pandas as pd
import cv2
from PIL import Image
from skimage.feature import local_binary_pattern, hog, graycomatrix, graycoprops

def load_model():
    model = pickle.load(open('modelCombinedSemua2.pkl', 'rb'))
    return model
model = load_model()

def predict_disease(features):
    prediction = model.predict(features)
    return prediction

def isskin(image):
    # Convert PIL.Image to a NumPy array
    image_array = np.array(image)
    if image_array.shape[2] == 4:  # RGBA image
        image_array = cv2.cvtColor(image_array, cv2.COLOR_RGBA2RGB)
    elif len(image_array.shape) == 2:  # Grayscale image
        image_array = cv2.cvtColor(image_array, cv2.COLOR_GRAY2RGB)
    
    # Resize the image to match the model input size
    image_array = cv2.resize(image_array, (224, 224))
    
    # Normalize pixel values
    image_array = image_array / 255.0
    
    # Add a batch dimension
    image_array = np.expand_dims(image_array, axis=0)

    # Get current positions of trackbars
    h_min = 0
    h_max = 128
    s_min = 100
    s_max = 150
    v_min = 0
    v_max = 128

    # Convert the image to HSV color space
    # hsv_image = cv2.cvtColor(image_array, cv2.COLOR_RGB2HSV)
    hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

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
    result = cv2.merge((skin, alpha))

    cv2.imwrite("skin_detection_result.png", result)
    return result

def extract(img):
    if img is None:
        print(f"Failed to load {img}. Skipping.")
        return None
    skin_img = isskin(img)

    if skin_img.shape[2] == 4:
        skin_img = skin_img[:, :, :3] 
    denoised = cv2.fastNlMeansDenoising(skin_img, h=10)

    img_resized = cv2.resize(denoised, (128, 128))

    # Convert to grayscale
    gray = cv2.cvtColor(img_resized, cv2.COLOR_RGB2GRAY)

    ## BARU
    lbp = local_binary_pattern(gray, P=8, R=1, method='uniform')
    hist, _ = np.histogram(lbp, bins=np.arange(0, 11), range=(0, 3))
    hist = hist.astype('float')
    hist /= (hist.sum() + 1e-6)
    lbp_features = hist

    hog_features, _ = hog(gray, orientations=9, pixels_per_cell=(8,8), cells_per_block=(2,2), block_norm='L2-Hys', visualize=True)

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hist_h = cv2.calcHist([hsv], [0], None, [256], (0, 128))
    hist_s = cv2.calcHist([hsv], [1], None, [256], (100, 150))
    hist_v = cv2.calcHist([hsv], [2], None, [256], (0, 128))
    hist_h /= hist_h.sum() + 1e-6
    hist_s /= hist_s.sum() + 1e-6
    hist_v /= hist_v.sum() + 1e-6
    color_features = np.concatenate([hist_h.flatten(), hist_s.flatten(), hist_v.flatten()])

    glcm = graycomatrix(gray, distances=[5], angles=[0], levels=256, symmetric=True, normed=True)
    glcm_features = []
    properties = ('contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation')
    for prop in properties:
        glcm_features.append(graycoprops(glcm, prop).flatten())

    return np.concatenate([lbp_features.flatten(), hog_features.flatten(), color_features.flatten(), glcm_features.flatten()])

st.title('Face Skin Disease Recognition\n(Acne / Eyebags / Redness)')

input_option = st.radio("Choose an image source:", ["Gallery", "Camera"])
selected_image = None

if input_option == "Gallery":
    uploaded_image = st.file_uploader("Upload an image from your device or use the camera to capture one:", 
                                type=["jpg", "jpeg", "png"], 
                                accept_multiple_files=False, 
                                label_visibility="visible")
    if uploaded_image is not None:
        selected_image = Image.open(uploaded_image)
        st.success("Image uploaded successfully...")
elif input_option == "Camera":
    camera_image = st.camera_input("Capture an image using your camera:")
    if camera_image is not None:
        selected_image = Image.open(camera_image)
        st.success("Image captured successfully...")

# Submit button
if st.button("Submit"):
    if selected_image:
        st.image(selected_image, caption="Submitted Image", use_column_width=True)
        
        # TEST
        temp_image = cv2.imread('./Input/red girl.jpg')

        # Preprocess the image
        extracted_data = extract(temp_image)
        
        # Make predictions
        predictions = model.predict(extracted_data)
        
        # Display predictions
        st.write("Prediction Results:")
        st.write(predictions)
    else:
        st.warning("Please upload or capture an image before submitting.")