import joblib
import numpy as np
import cv2
import PIL
from skimage.feature import local_binary_pattern
import torch
import torch.nn as nn
from torchvision import models
from torchvision import transforms
from PIL import Image
import sys

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def extract_features(image_path):

    img = cv2.imread(image_path)
    if img is None:
        print("Could not load image:", image_path)
        return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    features = []

    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    features.append(lap_var)

    edges = cv2.Canny(gray, 100, 200)
    edge_density = np.sum(edges > 0) / edges.size
    features.append(edge_density)

    glare_ratio = np.sum(gray > 240) / gray.size
    features.append(glare_ratio)

    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)

    magnitude = np.abs(fshift)

    h, w = magnitude.shape
    center_h, center_w = h // 2, w // 2

    radius = min(h, w) // 8

    mask = np.ones_like(magnitude)
    mask[center_h-radius:center_h+radius,
         center_w-radius:center_w+radius] = 0

    high_freq_energy = np.mean(magnitude * mask)
    features.append(high_freq_energy)

    lbp = local_binary_pattern(
        gray,
        P=8,
        R=1,
        method="uniform"
    )

    hist, _ = np.histogram(
        lbp.ravel(),
        bins=np.arange(0, 11),
        density=True
    )

    features.extend(hist)

    return np.array(features)


def load_effmodel():
    eff_model = models.efficientnet_b0(weights=None)

    in_features = eff_model.classifier[1].in_features

    eff_model.classifier = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(in_features, 2)
    )

    eff_model.load_state_dict(
        torch.load("enet.pth", map_location=device)
    )

    eff_model = eff_model.to(device)
    eff_model.eval()

    print("EfficientNet loaded!")

    return eff_model

def load_xg():
    xgb_model = joblib.load("model.pkl")
    print("XGBoost model loaded!")

    return xgb_model

def preprocessing():
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    return val_transform


def predict(img_path):
    image = Image.open(img_path).convert("RGB")
    if image is None:
        print("Image not loaded")
        return
       
    val_transform = preprocessing()
    x = val_transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        eff_model = load_effmodel()
        output = eff_model(x)
        eff_score = torch.softmax(output, dim=1)[0][1].item()

    xgb_model = load_xg()
    feats = extract_features(img_path).reshape(1, -1)
    xgb_score = xgb_model.predict_proba(feats)[0][1]

    final_score = 0.6 * eff_score + 0.4 * xgb_score

    prediction = "SCREEN" if final_score > 0.5 else "REAL"


    print(f"Prediction         : {prediction}")

    return final_score


def main():
    image_path = sys.argv[1]
    predict(image_path)
    

main()