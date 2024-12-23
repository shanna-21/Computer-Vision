import streamlit as st
import pickle
import numpy as np
import pandas as pd
import os
import cv2
from PIL import Image
from skimage.feature import local_binary_pattern, hog, graycomatrix, graycoprops

def load_model():
    model = pickle.load(open('cana.pkl', 'rb'))
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
    
    denoised = cv2.fastNlMeansDenoising(img, h=10)
    img_resized = cv2.resize(denoised, (128, 128))
    
    skin_img = isskin(img_resized)

    if skin_img.shape[2] == 4:
        skin_img = skin_img[:, :, :3] 
    
    # Convert to grayscale
    gray = cv2.cvtColor(skin_img, cv2.COLOR_RGB2GRAY)

    ## BARU
    lbp = local_binary_pattern(gray, P=8, R=1, method='uniform')
    hist, _ = np.histogram(lbp, bins=np.arange(0, 11), range=(0, 3))
    hist = hist.astype('float')
    hist /= (hist.sum() + 1e-6)
    lbp_features = hist.reshape(1, -1)

    hog_features, _ = hog(gray, orientations=9, pixels_per_cell=(8,8), cells_per_block=(2,2), block_norm='L2-Hys', visualize=True)
    hog_features = hog_features.reshape(1, -1)

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hist_h = cv2.calcHist([hsv], [0], None, [256], (0, 128))
    hist_s = cv2.calcHist([hsv], [1], None, [256], (100, 150))
    hist_v = cv2.calcHist([hsv], [2], None, [256], (0, 128))
    hist_h /= hist_h.sum() + 1e-6
    hist_s /= hist_s.sum() + 1e-6
    hist_v /= hist_v.sum() + 1e-6
    color_features = np.concatenate([hist_h.flatten(), hist_s.flatten(), hist_v.flatten()])
    color_features = color_features.reshape(1, -1)

    glcm = graycomatrix(gray, distances=[5], angles=[0], levels=256, symmetric=True, normed=True)
    glcm_features = []
    properties = ('contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation')
    for prop in properties:
        # glcm_features.append(graycoprops(glcm, prop).flatten())
        glcm_features.append(graycoprops(glcm, prop).flatten())
    #     prop_features = graycoprops(glcm, prop).flatten()
    #     glcm_features.append(prop_features)
    # glcm_features = np.concatenate(glcm_features)
    glcm_features = np.array([graycoprops(glcm, prop).flatten() for prop in properties])
    glcm_features = glcm_features.flatten().reshape(1, -1)
    

    return np.concatenate([lbp_features, hog_features, color_features, glcm_features], axis=1)
    
    # kalo pake kent.pkl
    # features = np.concatenate([lbp_features, hog_features, color_features, glcm_features], axis=1)
    # if features.shape[1] > 8878:
    #     features = features[:, :8878]
    
    # return features


st.title('Face Skin Disease Recognition\n(Acne / Eyebags / Redness)')

input_option = st.radio("Choose an image source:", ["Gallery", "Camera"])
selected_image = None

image_dir = "uploaded_images"
os.makedirs(image_dir, exist_ok=True)

if input_option == "Gallery":
    uploaded_image = st.file_uploader("Upload an image from your device or use the camera to capture one:", 
                                type=["jpg", "jpeg", "png"], 
                                accept_multiple_files=False, 
                                label_visibility="visible")
    if uploaded_image is not None:
        # selected_image = Image.open(uploaded_image)
        # st.success("Image uploaded successfully...")
        # Save the uploaded image to a file
        image_path = os.path.join(image_dir, uploaded_image.name)
        with open(image_path, "wb") as f:
            f.write(uploaded_image.getbuffer())
        selected_image = Image.open(image_path)
        st.success("Image uploaded successfully...")
        # st.image(selected_image, caption="Uploaded Image")
        st.write(f"Image saved at: {image_path}")
elif input_option == "Camera":
    camera_image = st.camera_input("Capture an image using your camera:")
    if camera_image is not None:
        # selected_image = Image.open(camera_image)
        # st.success("Image captured successfully...")
        # Save the captured image to a file
        image_path = os.path.join(image_dir, "captured_image.png")  # You can change the name if needed
        with open(image_path, "wb") as f:
            f.write(camera_image.getbuffer())
        selected_image = Image.open(image_path)
        st.success("Image captured successfully...")
        # st.image(selected_image, caption="Captured Image")
        st.write(f"Image saved at: {image_path}")

# Submit button
if st.button("Submit"):
    if selected_image:
        st.image(selected_image, caption="Submitted Image", use_column_width=True)
        
        # TEST
        temp_image = cv2.imread(image_path)

        # Preprocess the image
        extracted_data = extract(temp_image)
        
        # Make predictions
        predictions = model.predict(extracted_data)
        
        # Display predictions
        st.write("Prediction Results:")
        if predictions == 0:
            st.write("Acne")
            st.write("What you can do if you have acne")
            st.markdown("""
            - Wash your face twice a day and after sweating
            - Stop scrubbing your face and other acne-prone skin
            - Resist touching, picking, and popping your acne
            - Properly wash your face
            - Stay hydrated
            - Limit makeup
            - Try not to touch face
            - Limit sun exposure
                        """)
        elif predictions == 2:
            st.write("Redness")
            st.write("What you can do if you have redness")
            st.markdown("""
            - Avoid harsh skincare products (like those with alcohol or fragrance).
            - Use gentle, fragrance-free cleansers and moisturizers.
            - Protect your skin from the sun with sunscreen (SPF 30 or higher).
            - Apply cool compresses to calm redness.
            - Avoid hot showers, spicy foods, and alcohol, which can trigger redness.
            - Stay hydrated and maintain a healthy diet.
            - Use makeup with green undertones to neutralize redness.
            - Reduce stress, as it can exacerbate redness.
                        """)
        elif predictions == 1:
            st.write("Eyebags")
            st.write("What you can do if you have eyebags")
            st.markdown("""
            - Use a cool compress on your eyes
            - Make sure you get enough sleep
            - Sleep with your head raised slightly
            - Try to avoid drinking fluids before bed
            - Limit salt in your diet
            - Try to reduce your allergy symptoms
            - Don't smoke
            - Wear sunscreen every day
                        """)
        
    else:
        st.warning("Please upload or capture an image before submitting.")