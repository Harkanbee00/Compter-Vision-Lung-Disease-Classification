import os
import streamlit as st
import pandas as pd
from tensorflow.keras.models import load_model
from PIL import Image, ImageOps, UnidentifiedImageError
import numpy as np

# This must be the first Streamlit command
st.set_page_config(page_title="Image Classifier App", page_icon="🫁", layout="centered")

@st.cache_resource
def load_teachable_model():
    model = load_model("model.savedmodel", compile=False)
    # Strip any trailing newlines from class names
    class_names = [line.strip() for line in open("labels.txt", "r").readlines()]
    return model, class_names

def predict_image(image, model, class_names):
    """
    Transforms the image to fit the teachable machine model requirements,
    and returns the predicted label, confidence, and all probabilities.
    """
    # Resize the image to 224x224, as expected by the model
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    
    # Convert image to numpy array
    image_array = np.asarray(image)
    
    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
    
    # Create the input shape for the model
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized_image_array
    
    # Run prediction
    prediction = model.predict(data)
    
    # Get the predicted label index
    index = np.argmax(prediction)
    class_name = class_names[index]
    
    # Normally class names from teachable machine are like "0 ClassName", so we strip the prefix if it exists
    if " " in class_name and class_name.split(" ")[0].isdigit():
        class_name = class_name.split(" ", 1)[1]
        
    confidence = prediction[0][index]
    return class_name, confidence, prediction[0]

# --- SIDEBAR NAVIGATION ---
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to:", ["Upload Image", "Sample Images"])

if page == "Upload Image":
    st.title("🫁 Lung Disease Image Classifier")
    st.markdown("Upload an image and get a prediction from a trained model.")

    # Add an informational note so your audience sees the model limitation.
    st.info("This app is for educational purposes only. It is not a medical device and must not be used as a clinical diagnosis tool.", icon="ℹ️")

    # Safely check if files exist before trying to load them
    if not os.path.exists("model.savedmodel"):
        st.error("Model directory 'model.savedmodel' not found! Please add it to the same folder as this app.")
        st.stop()
    if not os.path.exists("labels.txt"):
        st.error("Labels file 'labels.txt' not found! Please add it to the same folder as this app.")
        st.stop()

    model, class_names = load_teachable_model()

    uploaded_file = st.file_uploader("Upload an X-ray image", type=["jpg", "jpeg", "png"])

    # Run the prediction block only after a file has been uploaded.
    if uploaded_file is not None:
        # Try to open the uploaded file as an image.
        try:
            # Read the uploaded file into a PIL image object. 
            # .convert("RGB") ensures images like grayscale or PNG with transparency have 3 channels based on what the model expects
            uploaded_image = Image.open(uploaded_file).convert("RGB")
        # Catch errors when the uploaded file is not a valid image.
        except UnidentifiedImageError:
            # Show a helpful error message in the app.
            st.error("The uploaded file could not be read as an image. Please upload a JPG, JPEG, or PNG file.")
            # Stop further processing for this run.
            st.stop()

        # Show the uploaded image on the page so the user can confirm it.
        st.image(uploaded_image, caption="Uploaded image", use_container_width=True)

        # Add a button so the user can control when inference runs.
        if st.button("Run prediction"):
            # Run the model on the uploaded image and collect the outputs.
            predicted_label, confidence, probabilities = predict_image(uploaded_image, model, class_names)

            # Write a section heading for the result.
            st.subheader("Prediction result")

            # Show a red box when the predicted class is Lung Opacity or Viral Pneumonia.
            if predicted_label in ["Lung_Opacity", "Viral Pneumonia"]:
                # Display the final predicted label in an error-style box.
                st.error(f"{predicted_label} ({confidence:.1%})")
            # Show a green box when the predicted class is normal.
            else:
                # Display the final predicted label in a success-style box.
                st.success(f"{predicted_label} ({confidence:.1%})")

            # Strip prefixes from class names for probability chart display
            display_names = [name.split(" ", 1)[1] if " " in name and name.split(" ")[0].isdigit() else name for name in class_names]

            # Build a small DataFrame to display class probabilities neatly.
            probability_frame = pd.DataFrame(
                {
                    "Class": display_names,
                    "Probability (%)": [float(p) * 100 for p in probabilities],
                }
            )

            # Add a heading for the probability chart.
            st.subheader("Prediction Probabilities")
            # Show a bar chart of the class probabilities.
            st.bar_chart(probability_frame.set_index("Class"))
    # Show a helpful message before any file is uploaded.
    else:
        # Prompt the user to upload an image to begin.
        st.info("Upload a Lung X-ray image to start the demo.")

elif page == "Sample Images":
    st.title("📸 Sample X-Rays")
    st.markdown("Select a sample image below to run a prediction instantly.")
    
    st.divider()
    
    # We will look for images inside a folder named 'sample_images'
    samples_dir = "Sample Images"
    
    if not os.path.exists(samples_dir):
        st.warning(f"No '{samples_dir}' folder found. Please properly create a folder named '{samples_dir}' inside your repository and place your sample images inside it.")
    else:
        # Get all image files in the directory
        valid_extensions = (".jpg", ".jpeg", ".png")
        sample_files = [f for f in os.listdir(samples_dir) if f.lower().endswith(valid_extensions)]
        
        if not sample_files:
            st.info(f"The '{samples_dir}' folder is currently empty! Please place some .jpg or .png X-ray images inside it.")
        else:
            # Load the model directly here too
            model, class_names = load_teachable_model()

            # Display the images in a grid format with 3 columns
            cols = st.columns(3)
            for index, image_name in enumerate(sample_files):
                image_path = os.path.join(samples_dir, image_name)
                try:
                    img = Image.open(image_path).convert("RGB")
                    # Route image rendering into a 3-column grid structure
                    with cols[index % 3]:
                        st.image(img, caption=image_name, use_container_width=True)
                        
                        # Add a quick predict button underneath the image
                        if st.button("Predict this Image", key=f"predict_{index}"):
                            # Run the prediction
                            predicted_label, confidence, probabilities = predict_image(img, model, class_names)
                            
                            # Show a red box when the predicted class is Lung Opacity or Viral Pneumonia.
                            if predicted_label in ["Lung_Opacity", "Viral Pneumonia"]:
                                st.error(f"{predicted_label} ({confidence:.1%})")
                            # Show a green box when the predicted class is normal.
                            else:
                                st.success(f"{predicted_label} ({confidence:.1%})")
                except Exception as e:
                    st.error(f"Could not load image {image_name}")
