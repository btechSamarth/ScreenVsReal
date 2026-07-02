import os
import tempfile
import joblib
import numpy as np
import cv2
from skimage.feature import local_binary_pattern
import torch
import torch.nn as nn
from torchvision import models
from torchvision import transforms
from PIL import Image
import streamlit as st


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def extract_features(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()

    edges = cv2.Canny(gray, 100, 200)
    edge_density = np.sum(edges > 0) / edges.size

    glare_ratio = np.sum(gray > 240) / gray.size

    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    magnitude = np.abs(fshift)

    h, w = magnitude.shape
    ch, cw = h // 2, w // 2
    r = min(h, w) // 8

    mask = np.ones_like(magnitude)
    mask[ch-r:ch+r, cw-r:cw+r] = 0
    high_freq_energy = np.mean(magnitude * mask)

    lbp = local_binary_pattern(gray, 8, 1, method="uniform")
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 11), density=True)

    feature_vector = [lap_var, edge_density, glare_ratio, high_freq_energy]
    for val in lbp_hist:
        feature_vector.append(val)

    return np.array(feature_vector)


@st.cache_resource
def load_effmodel():
    model = models.efficientnet_b0(weights=None)
    in_feats = model.classifier[1].in_features
    model.classifier = nn.Sequential(nn.Dropout(0.5), nn.Linear(in_feats, 2))

    weights_path = os.path.join(BASE_DIR, "enet.pth")
    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict)

    model.to(device)
    model.eval()

    return model


@st.cache_resource
def load_xg():
    return joblib.load(os.path.join(BASE_DIR, "model.pkl"))


@st.cache_resource
def get_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

def predict(img_path):
    image = Image.open(img_path).convert("RGB")

    val_transform = get_transform()
    x = val_transform(image).unsqueeze(0).to(device)

    eff_model = load_effmodel()
    with torch.no_grad():
        output = eff_model(x)
        eff_score = torch.softmax(output, dim=1)[0][1].item()

    xgb_model = load_xg()
    feats = extract_features(img_path)
    if feats is None:
        raise ValueError("Could not extract features from image.")
    feats = feats.reshape(1, -1)
    xgb_score = xgb_model.predict_proba(feats)[0][1]

    final_score = 0.6 * eff_score + 0.4 * xgb_score
    prediction = "SCREEN" if final_score > 0.5 else "REAL"

    return prediction, final_score, eff_score, xgb_score

st.set_page_config(page_title="Screen vs Real Image Detector", page_icon="📷")

st.title("📷 Screen vs Real Image Detector")
st.write("Upload an image to check whether it was captured from a real scene or photographed off a screen.")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)

    suffix = os.path.splitext(uploaded_file.name)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    try:
        with st.spinner("Running prediction..."):
            prediction, final_score, eff_score, xgb_score = predict(tmp_path)

        if prediction == "SCREEN":
            st.error(f"Prediction: **{prediction}**")
        else:
            st.success(f"Prediction: **{prediction}**")

        st.metric("Final Score", f"{final_score:.4f}")

        with st.expander("Details"):
            st.write(f"EfficientNet score: {eff_score:.4f}")
            st.write(f"XGBoost score: {xgb_score:.4f}")

    except Exception as e:
        st.error(f"Something went wrong during prediction: {e}")

    finally:
        os.remove(tmp_path)