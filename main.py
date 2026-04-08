import streamlit as st
from PIL import Image
from skin_disease.main import SkinDiseaseDetector


st.title("AI Skin Disease Detection")

st.write("Upload a skin lesion image for AI analysis")

uploaded_file = st.file_uploader("Upload Image", type=["jpg","png","jpeg"])

detector = SkinDiseaseDetector()

if uploaded_file:

    image = Image.open(uploaded_file)

    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Detect Disease"):

        result = detector.predict(image)

        st.success("Analysis Complete")

        st.write("### Result")
        st.write("Disease:", result["disease_name"])
        st.write("Confidence:", f"{result['confidence']:.2f}%")