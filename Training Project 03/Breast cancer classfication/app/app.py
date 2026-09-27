import os

import numpy as np
import streamlit as st

from PIL import Image

import tensorflow as tf


# --------------------------------------------------
# Configuration
# --------------------------------------------------

IMG_SIZE = (96, 96)

MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "models",
    "breast_cancer_classifier.keras"
)


# --------------------------------------------------
# Load model
# --------------------------------------------------

@st.cache_resource
def load_model():

    model = tf.keras.models.load_model(
        MODEL_PATH
    )

    return model


model = load_model()


# --------------------------------------------------
# Prediction function
# --------------------------------------------------

def predict_image(image):

    image = image.convert("RGB")

    image = image.resize(IMG_SIZE)

    image_array = np.array(
        image
    ).astype(
        np.float32
    ) / 255.0

    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    probability = model.predict(
        image_array,
        verbose=0
    )[0][0]

    if probability >= 0.5:

        prediction = "IDC"

    else:

        prediction = "Non-IDC"

    return prediction, float(probability)


# --------------------------------------------------
# Streamlit UI
# --------------------------------------------------

st.set_page_config(
    page_title="Breast Cancer Classification",
    page_icon="🔬",
    layout="centered"
)


st.title(
    "🔬 Breast Cancer Classification"
)

st.write(
    """
    Upload a breast histopathology image patch
    for model-based classification.
    """
)


uploaded_file = st.file_uploader(
    "Upload image",
    type=[
        "png",
        "jpg",
        "jpeg"
    ]
)


if uploaded_file is not None:

    image = Image.open(
        uploaded_file
    )

    st.image(
        image,
        caption="Uploaded Image",
        width="300"
    )

    if st.button(
        "Classify Image"
    ):

        prediction, probability = predict_image(
            image
        )

        st.subheader(
            "Prediction"
        )

        st.write(
            prediction
        )

        st.write(
            f"Model probability: {probability:.4f}"
        )

        st.info(
            """
            This application is an educational
            machine-learning demonstration and
            must not be used for medical diagnosis.
            """
        )