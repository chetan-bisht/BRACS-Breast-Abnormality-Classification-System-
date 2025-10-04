import streamlit as st
import os
from PIL import Image
import numpy as np
import tensorflow as tf
import pandas as pd

# --- CORRECT IMPORT FOR RESNET50 ---
from tensorflow.keras.applications.resnet50 import preprocess_input

# --- PAGE CONFIGURATION ---
# This must be the first Streamlit command in your script.
st.set_page_config(
    page_title="Mammogram Patch Abnormality Classifier",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- CONFIGURATION ---
# Use forward slashes for paths for better cross-platform compatibility
MODEL_PATH = "C:/breast-cancer-detection/DDSM_Classifier_Dashboard/BEST_DDSM_MODEL.h5"
ASSETS_PATH = "C:/breast-cancer-detection/DDSM_Classifier_Dashboard/assets"
CLASS_NAMES = ["Calcification", "Mass", "Normal"]
IMAGE_INPUT_SIZE = (224, 224)


# --- STYLING ---
# Inject custom CSS for a more polished look
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# You can create a style.css file or embed it directly
st.markdown("""
<style>
    /* Main app background */
    .stApp {
        /* background-color: #f0f2f6; /* Use if you want a fixed background color */
    }

    /* Card-like containers */
    [data-testid="stVerticalBlock"] > [style*="flex-direction: column;"] > [data-testid="stVerticalBlock"] {
        border: 1px solid rgba(0, 0, 0, 0.1);
        border-radius: 15px;
        padding: 20px;
        box-shadow: 0 4px 8px 0 rgba(0,0,0,0.1);
        transition: 0.3s;
        background-color: #ffffff; /* Light mode card background */
    }

    /* Ensure dark mode cards have a dark background */
    body.dark [data-testid="stVerticalBlock"] > [style*="flex-direction: column;"] > [data-testid="stVerticalBlock"] {
        background-color: #2C2C2C; /* Dark mode card background from config.toml */
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    /* Header styling */
    h1, h2, h3 {
        font-weight: 600;
    }

    /* Verdict card styling */
    .verdict-card {
        padding: 25px;
        border-radius: 15px;
        text-align: center;
        margin-bottom: 20px;
        border-width: 2px;
        border-style: solid;
    }
    .verdict-card h2 {
        font-size: 2.5em;
        margin: 0;
        color: white; /* Text color for all verdict cards */
    }
    .verdict-card.normal {
        border-color: #28a745;
        background-color: #28a745;
    }
    .verdict-card.calcification {
        border-color: #ffc107;
        background-color: #ffc107;
    }
    .verdict-card.mass {
        border-color: #dc3545;
        background-color: #dc3545;
    }

</style>
""", unsafe_allow_html=True)


# --- MODEL LOADING ---
@st.cache_resource
def load_keras_model(model_path):
    """Loads the Keras model from disk, handling potential errors."""
    st.write("Cache miss: Loading Keras model...")
    try:
        model = tf.keras.models.load_model(
            model_path,
            # The custom_objects dictionary is often not needed if preprocess_input
            # is just a function and not a custom layer. Let's keep it safe.
            custom_objects={'preprocess_input': preprocess_input}
        )
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.error("Please ensure the model path is correct and the file is not corrupted.")
        return None

# --- PREDICTION FUNCTION ---
def get_prediction_probabilities(model, image: Image.Image):
    """Preprocesses an image and returns model prediction probabilities."""
    image = image.convert("RGB").resize(IMAGE_INPUT_SIZE)
    image_array = np.array(image)
    input_batch = np.expand_dims(image_array, axis=0)
    processed_batch = preprocess_input(input_batch)
    return model.predict(processed_batch)

# --- UI HELPER FUNCTIONS ---
def display_verdict_card(predicted_class, confidence):
    """Displays a styled card with the final verdict."""
    icon_map = {"Normal": "✅", "Calcification": "⚠️", "Mass": "❗️"}
    css_class_map = {"Normal": "normal", "Calcification": "calcification", "Mass": "mass"}
    
    verdict_html = f"""
    <div class="verdict-card {css_class_map.get(predicted_class, '')}">
        <h2>{icon_map.get(predicted_class, '')} {predicted_class}</h2>
    </div>
    """
    st.markdown(verdict_html, unsafe_allow_html=True)
    st.metric(label="Model Confidence", value=f"{confidence:.2%}")

def display_welcome_gallery(assets_path):
    """Displays the initial welcome page with an image gallery."""
    st.markdown(
        "Welcome! This tool uses a **ResNet50** model to classify mammogram images. "
        "Upload an image in the sidebar to try the live classifier. Below are some example results and performance metrics from the model's training."
    )
    st.markdown("---")

    asset_files = []
    if os.path.exists(assets_path) and os.path.isdir(assets_path):
        asset_files = sorted([f for f in os.listdir(assets_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

    if asset_files:
        st.subheader("Model Results & Metrics Gallery")
        # Create a grid of 3 columns
        cols = st.columns(3)
        for i, file_name in enumerate(asset_files):
            # Place each image in a column, cycling through columns 0, 1, 2
            with cols[i % 3]:
                with st.container(border=True):
                    image_path = os.path.join(assets_path, file_name)
                    st.image(image_path, caption=file_name.split('.')[0].replace('_', ' ').title(), use_container_width=True)
    else:
        st.warning(f"Could not find any example result images in the assets folder: `{assets_path}`")

def display_analysis_page(model, input_image):
    """Displays the analysis results for the uploaded image."""
    st.markdown(
        "This is an educational tool and **not a substitute for professional medical diagnosis.**"
    )
    
    col1, col2 = st.columns([0.6, 0.4]) # Give more space to the image
    
    with col1:
        with st.container(border=True):
            st.subheader("Your Input Image")
            st.image(input_image, use_container_width=True)

    with col2:
        with st.container(border=True):
            st.subheader("Analysis Results")
            with st.spinner("Model is analyzing the image..."):
                probabilities = get_prediction_probabilities(model, input_image)[0]
                top_prediction_index = np.argmax(probabilities)
                predicted_class = CLASS_NAMES[top_prediction_index]
                confidence = probabilities[top_prediction_index]
            
            # Display the custom verdict card
            display_verdict_card(predicted_class, confidence)
            
            # Display the probability chart
            st.write("##### Prediction Probabilities")
            prob_df = pd.DataFrame({'Class': CLASS_NAMES, 'Probability': probabilities})
            st.bar_chart(prob_df.set_index('Class'))

            with st.expander("View Detailed Probabilities"):
                for class_name, prob in zip(CLASS_NAMES, probabilities):
                    st.write(f"{class_name}: **{prob:.2%}**")
                    st.progress(float(prob))

# --- MAIN APP LOGIC ---
def main():
    """Main function to run the Streamlit app."""
    model = load_keras_model(MODEL_PATH)
    if model is None:
        st.stop() # Stop execution if the model failed to load

    # --- SIDEBAR ---
with st.sidebar:
    logo_path = os.path.join(ASSETS_PATH, "app_logo.png")
    if os.path.exists(logo_path):
        st.image(logo_path, width=150)
    st.header("🔬 Live Classifier")
    st.markdown("Upload your own mammogram image to get a live prediction.")
    uploaded_file = st.file_uploader(
        "Choose an image...",
        type=["png", "jpg", "jpeg"],
        label_visibility="collapsed"
    )
    st.markdown("---")
    st.info("**Theme Switcher:** Use the **Settings** menu (top right) to switch between light and dark modes.")

# --- MAIN PAGE ---
st.title("🩺 Breast Abnormality Classification System")

model = load_keras_model(MODEL_PATH)  # Define model here

if uploaded_file is not None:
    input_image = Image.open(uploaded_file)
    display_analysis_page(model, input_image)
else:
    display_welcome_gallery(ASSETS_PATH)

# --- FOOTER ---
st.markdown("---")
st.markdown("Developed with ❤️ using Streamlit & TensorFlow. Not for clinical use.")