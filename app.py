import streamlit as st
from pathlib import Path
from PIL import Image, UnidentifiedImageError
from utils import predict_image, load_model

# Cache model loading for performance
@st.cache_resource
def get_model():
    """Load and cache the model"""
    return load_model()

def main():
    st.set_page_config(page_title="AI Skin Disease Detection", layout="centered")
    st.title("🧠 AI Skin Disease Detection System")
    st.write("Upload a skin image to detect possible skin conditions using AI.")
    st.warning("⚠️ This is not a medical diagnosis. Consult a doctor for professional advice.")

    # Load model once and cache it
    try:
        model = get_model()
    except Exception as e:
        st.error(f"❌ Failed to load model: {str(e)}")
        st.stop()

    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        try:
            image = Image.open(uploaded_file).convert("RGB")
        except UnidentifiedImageError:
            st.error("❌ Unable to read image. Please upload a valid JPG/PNG image.")
            return

        # Fix deprecated parameter
        st.image(image, caption="Uploaded Image", width="stretch")

        if st.button("🔍 Predict Disease", use_container_width=True):
            with st.spinner("🔄 Running inference..."):
                try:
                    # predict_image now returns (label, confidence) tuple
                    predicted_label, confidence_score = predict_image(image)
                except Exception as ex:
                    st.error(f"❌ Inference error: {str(ex)}")
                    return

            st.success(f"✅ Prediction: {predicted_label}")
            st.info(f"📊 Confidence: {confidence_score:.2f}%")
            
            if confidence_score < 50:
                st.warning("⚠️ Low confidence prediction. Please try another image for better accuracy.")
            
            st.markdown("---")
            st.info("💡 Always consult a dermatologist for professional medical advice.")


if __name__ == "__main__":
    main()