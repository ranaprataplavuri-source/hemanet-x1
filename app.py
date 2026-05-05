# app.py - Hema Net-X Batch Processing (Grad-CAM for ALL patients)

import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
from torchvision import transforms
import timm
import os
import pandas as pd
from datetime import datetime

# Download model from Google Drive if not exists
import gdown
import os

MODEL_PATH = 'models/hemanet_model.pth'
MODEL_URL = "https://drive.google.com/file/d/1BgRfJQb4ZfxYmLMAHM-ldWEus2vgCRoO/view?usp=sharing"  # Replace with your file ID

if not os.path.exists(MODEL_PATH):
    os.makedirs('models', exist_ok=True)
    with st.spinner("Downloading model (42 MB)... This may take a minute"):
        gdown.download(MODEL_URL, MODEL_PATH, quiet=False)
    st.success("✅ Model downloaded!")

# ============= PAGE CONFIGURATION =============
st.set_page_config(
    page_title="Hema Net-X - Multi-Patient Batch Analysis",
    page_icon="🩸",
    layout="wide"
)

st.title("🩸 Hema Net-X: Multi-Patient Blood Cell Analysis System")
st.markdown("Upload up to **10 patient images** at once for batch analysis with **CSV export**")

# ============= SIDEBAR SETTINGS =============
with st.sidebar:
    st.header("⚙️ Settings")
    
    task_filter = st.selectbox(
        "Filter Results",
        ["Show All", "Blood Cell Classification", "Leukemia Detection", "Malaria Detection"]
    )
    
    confidence_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.5)
    show_gradcam = st.checkbox("Show Grad-CAM for all patients", value=False)
    
    st.markdown("---")
    st.header("📊 Batch Processing")
    st.info("""
    **Features:**
    - Process up to 10 images at once
    - Export results as CSV
    - Patient ID tracking
    - Confidence scores for all tasks
    - Grad-CAM for all patients
    """)

# ============= MODEL ARCHITECTURE =============
class CNNBranch(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.Dropout2d(0.1),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.Dropout2d(0.1),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(), nn.Dropout2d(0.1),
            nn.AdaptiveAvgPool2d((7, 7))
        )
    def forward(self, x):
        return self.cnn(x)

class ViTBranch(nn.Module):
    def __init__(self):
        super().__init__()
        self.vit = timm.create_model('vit_tiny_patch16_224', pretrained=True, num_classes=0)
        self.projection = nn.Conv2d(192, 256, kernel_size=1)
        self.dropout = nn.Dropout(0.2)
    def forward(self, x):
        features = self.vit.forward_features(x)
        b, n, d = features.shape
        h = w = int(n**0.5)
        features = features[:, 1:, :].transpose(1, 2).reshape(b, d, h, w)
        return self.dropout(self.projection(features))

class EdgeBranch(nn.Module):
    def __init__(self):
        super().__init__()
        self.edge_cnn = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.Dropout2d(0.1),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.Dropout2d(0.1),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(), nn.Dropout2d(0.1),
            nn.AdaptiveAvgPool2d((7, 7))
        )
    def forward(self, x):
        return self.edge_cnn(x)

class HemaNetX(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn_branch = CNNBranch()
        self.vit_branch = ViTBranch()
        self.edge_branch = EdgeBranch()
        
        self.fusion = nn.Sequential(
            nn.Conv2d(256+256+256, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(), nn.Dropout2d(0.2),
            nn.Conv2d(512, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(), nn.Dropout2d(0.2),
            nn.AdaptiveAvgPool2d(1)
        )
        
        self.classifier_blood = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(256, 4)
        )
        self.classifier_leukemia = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(256, 2)
        )
        self.classifier_malaria = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(256, 2)
        )
        
    def forward(self, x):
        cnn_f = self.cnn_branch(x)
        vit_f = F.interpolate(self.vit_branch(x), size=(7,7), mode='bilinear')
        edge_f = self.edge_branch(x)
        
        fused = torch.cat([cnn_f, vit_f, edge_f], dim=1)
        fused = self.fusion(fused).squeeze(-1).squeeze(-1)
        
        return (self.classifier_blood(fused),
                self.classifier_leukemia(fused),
                self.classifier_malaria(fused))

# ============= CLASS NAMES =============
blood_classes = ['EOSINOPHIL', 'LYMPHOCYTE', 'MONOCYTE', 'NEUTROPHIL']
leukemia_classes = ['ALL (Leukemia)', 'Healthy']
malaria_classes = ['Parasitized', 'Uninfected']
blood_class_to_idx = {cls: idx for idx, cls in enumerate(blood_classes)}

# ============= TRANSFORM =============
def get_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

# ============= LOAD MODEL =============
@st.cache_resource
def load_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = HemaNetX().to(device)
    
    model_paths = ['models/hemanet_model.pth', 'hemanet_model.pth', './models/hemanet_model.pth', 'best_hemanetx.pth']
    for model_path in model_paths:
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location=device))
            st.success(f"✅ Loaded model from {model_path}")
            break
    
    model.eval()
    return model, device

# ============= BATCH PREDICTION =============
def predict_batch(images, patient_ids, model, device, transform):
    results = []
    
    if not images:
        return pd.DataFrame()
    
    batch_tensors = []
    valid_images = []
    valid_patient_ids = []
    
    for i, img in enumerate(images):
        try:
            if isinstance(img, str):
                img = Image.open(img).convert('RGB')
            else:
                img = img.convert('RGB')
            
            batch_tensors.append(transform(img))
            valid_images.append(img)
            valid_patient_ids.append(patient_ids[i] if i < len(patient_ids) else f"Patient_{i+1}")
        except Exception as e:
            st.warning(f"Could not process image: {e}")
    
    if not batch_tensors:
        return pd.DataFrame()
    
    batch_input = torch.stack(batch_tensors).to(device)
    
    with torch.no_grad():
        blood_out, leuk_out, mal_out = model(batch_input)
    
    for i, (img, patient_id) in enumerate(zip(valid_images, valid_patient_ids)):
        blood_probs = F.softmax(blood_out[i], dim=0).cpu().numpy()
        leuk_probs = F.softmax(leuk_out[i], dim=0).cpu().numpy()
        mal_probs = F.softmax(mal_out[i], dim=0).cpu().numpy()
        
        results.append({
            'patient_id': patient_id,
            'blood_prediction': blood_classes[np.argmax(blood_probs)],
            'blood_confidence': float(np.max(blood_probs)),
            'leukemia_prediction': leukemia_classes[np.argmax(leuk_probs)],
            'leukemia_confidence': float(np.max(leuk_probs)),
            'malaria_prediction': malaria_classes[np.argmax(mal_probs)],
            'malaria_confidence': float(np.max(mal_probs)),
            'blood_eosinophil': float(blood_probs[0]),
            'blood_lymphocyte': float(blood_probs[1]),
            'blood_monocyte': float(blood_probs[2]),
            'blood_neutrophil': float(blood_probs[3]),
            'leukemia_all': float(leuk_probs[0]),
            'leukemia_healthy': float(leuk_probs[1]),
            'malaria_parasitized': float(mal_probs[0]),
            'malaria_uninfected': float(mal_probs[1]),
            'overall_confidence': (float(np.max(blood_probs)) + float(np.max(leuk_probs)) + float(np.max(mal_probs))) / 3
        })
    
    return pd.DataFrame(results)

# ============= GRAD-CAM =============
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_full_backward_hook(self.save_gradient)
    
    def save_activation(self, module, input, output):
        self.activations = output.detach()
    
    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()
    
    def generate(self, input_image, target_class_idx=None):
        blood_out, leuk_out, mal_out = self.model(input_image.unsqueeze(0))
        model_output = blood_out
        
        if target_class_idx is None:
            target_class_idx = model_output[0].argmax().item()
        
        self.model.zero_grad()
        one_hot = torch.zeros((1, model_output.size(-1)))
        one_hot[0][target_class_idx] = 1
        model_output.backward(gradient=one_hot.to(next(self.model.parameters()).device), retain_graph=True)
        
        gradients = self.gradients.cpu().numpy()[0]
        activations = self.activations.cpu().numpy()[0]
        weights = np.mean(gradients, axis=(1, 2))
        
        cam = np.zeros(activations.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i]
        
        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (224, 224))
        
        if cam.max() - cam.min() > 1e-8:
            cam = (cam - cam.min()) / (cam.max() - cam.min())
        
        return cam, target_class_idx

# ============= MAIN =============
tab1, tab2 = st.tabs(["📤 Multi-Patient Upload", "📊 Results Dashboard"])

with tab1:
    st.subheader("📤 Upload Multiple Patient Images")
    
    num_patients = st.number_input("Number of patients to analyze (1-10)", min_value=1, max_value=10, value=3)
    
    cols = st.columns(min(3, num_patients))
    uploaded_files = []
    patient_ids = []
    
    for i in range(num_patients):
        col_idx = i % 3
        with cols[col_idx]:
            st.markdown(f"**Patient {i+1}**")
            uploaded = st.file_uploader(
                f"Upload image for Patient {i+1}", 
                type=['jpg', 'jpeg', 'png', 'bmp'],
                key=f"patient_{i}",
                label_visibility="collapsed"
            )
            if uploaded:
                uploaded_files.append(uploaded)
                patient_id = st.text_input(f"Patient ID", value=f"P{i+1:03d}", key=f"id_{i}", label_visibility="collapsed")
                patient_ids.append(patient_id)
            else:
                patient_ids.append(f"P{i+1:03d}")
    
    if uploaded_files and st.button("🔬 Analyze All Patients", type="primary", use_container_width=True):
        with st.spinner(f"🔄 Analyzing {len(uploaded_files)} patients..."):
            model, device = load_model()
            transform = get_transform()
            
            images = [Image.open(f).convert('RGB') for f in uploaded_files]
            results_df = predict_batch(images, patient_ids[:len(images)], model, device, transform)
            
            st.session_state['batch_results'] = results_df
            st.session_state['batch_images'] = images
            
            st.success(f"✅ Analysis complete for {len(results_df)} patients!")
            
            # Results Table
            st.subheader("📊 Analysis Results Table")
            display_cols = ['patient_id', 'blood_prediction', 'blood_confidence',
                           'leukemia_prediction', 'leukemia_confidence',
                           'malaria_prediction', 'malaria_confidence']
            st.dataframe(results_df[display_cols].style.format({
                'blood_confidence': '{:.1%}',
                'leukemia_confidence': '{:.1%}',
                'malaria_confidence': '{:.1%}'
            }), use_container_width=True)
            
            # Alerts
            st.subheader("⚠️ Alerts Summary")
            col1, col2 = st.columns(2)
            leuk_positive = results_df[results_df['leukemia_prediction'] == 'ALL (Leukemia)'].shape[0]
            mal_positive = results_df[results_df['malaria_prediction'] == 'Parasitized'].shape[0]
            with col1:
                if leuk_positive > 0:
                    st.error(f"🚨 {leuk_positive} patient(s) show signs of ALL Leukemia")
                else:
                    st.success("No leukemia detected")
            with col2:
                if mal_positive > 0:
                    st.error(f"🚨 {mal_positive} patient(s) show signs of Malaria")
                else:
                    st.success("No malaria detected")
            
            # ========== GRAD-CAM FOR ALL PATIENTS ==========
            if show_gradcam and images:
                st.subheader("🔥 Grad-CAM Visualization (All Patients)")
                
                # Create grid for all patients
                num_grad_cols = min(3, len(images))
                grad_cols = st.columns(num_grad_cols)
                
                for idx, (img, row) in enumerate(zip(images, results_df.iterrows())):
                    col_idx = idx % num_grad_cols
                    with grad_cols[col_idx]:
                        st.markdown(f"**Patient {idx+1}: {row[1]['patient_id']}**")
                        
                        try:
                            input_tensor = transform(img).unsqueeze(0).to(device)
                            pred_class = row[1]['blood_prediction']
                            target_idx = blood_class_to_idx.get(pred_class, 0)
                            
                            target_layer = model.fusion[-2]
                            grad_cam = GradCAM(model, target_layer)
                            heatmap, _ = grad_cam.generate(input_tensor.squeeze(0), target_idx)
                            
                            img_array = np.array(img.resize((224, 224)))
                            heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
                            heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
                            overlay = cv2.addWeighted(img_array, 0.6, heatmap_color, 0.4, 0)
                            
                            st.image(overlay, caption=f'Pred: {pred_class}', use_container_width=True)
                        except Exception as e:
                            st.warning(f"Grad-CAM failed: {e}")
                
                st.caption("🔴 Red = High focus | 🔵 Blue = Low focus")
            
            # Download
            csv = results_df.to_csv(index=False)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            st.download_button("📥 Download CSV", csv, f"hemanet_patients_{timestamp}.csv", "text/csv")

with tab2:
    st.subheader("📊 Results Dashboard")
    if 'batch_results' in st.session_state and not st.session_state['batch_results'].empty:
        results_df = st.session_state['batch_results']
        st.dataframe(results_df, use_container_width=True)
    else:
        st.info("No results yet. Upload and analyze patient images in the first tab!")

st.markdown("---")
st.caption("🩸 Hema Net-X - Multi-Patient Blood Cell Analysis System")
