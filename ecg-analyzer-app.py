import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import time

# Page configuration
st.set_page_config(
    page_title="ECG Analyzer",
    page_icon="❤️",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-title {
        text-align: center;
        font-size: 42px;
        color: #FF4B4B;
        margin-bottom: 20px;
    }
    .sub-title {
        text-align: center;
        font-size: 24px;
        color: #4B4B4B;
        margin-bottom: 30px;
    }
    .stProgress > div > div > div > div {
        background-color: #FF4B4B;
    }
    .result-text {
        font-size: 24px;
        font-weight: bold;
        padding: 20px;
        border-radius: 10px;
    }
    .instructions {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    .footer {
        text-align: center;
        margin-top: 40px;
        font-size: 14px;
        color: #888888;
    }
</style>
""", unsafe_allow_html=True)

# Title and introduction
st.markdown("<h1 class='main-title'>❤️ ECG Analyzer</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-title'>Upload an ECG image to analyze cardiac conditions</p>", unsafe_allow_html=True)

# Create two columns for layout
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("<div class='instructions'>", unsafe_allow_html=True)
    st.markdown("### 📋 Instructions")
    st.markdown("""
    1. Upload a clear PNG image of an ECG reading
    2. Wait for the AI to analyze the image
    3. Review the predicted condition and information
    
    The system can detect the following conditions:
    - Left Bundle Branch Block
    - Normal Heart Rhythm
    - Premature Atrial Contraction
    - Ventricular Fibrillation
    - Premature Ventricular Contraction
    - Right Bundle Branch Block
    """)
    st.markdown("</div>", unsafe_allow_html=True)
    
    # File uploader
    uploaded_file = st.file_uploader("Choose an ECG image", 
                                    type="png", 
                                    help="Upload a PNG image of an ECG reading")

# Load the model
@st.cache_resource
def load_ecg_model():
    model_path = 'ECG.h5'  # Replace with your model path
    return load_model(model_path)

try:
    model = load_ecg_model()
    model_loaded = True
except Exception as e:
    st.error(f"Error loading model: {e}")
    model_loaded = False

# Condition descriptions for education
condition_info = {
    "Left Bundle Branch Block": {
        "description": "Left Bundle Branch Block (LBBB) is a cardiac conduction abnormality where electrical impulses are delayed or blocked along the left bundle branch.",
        "indicators": "Wide QRS complex (≥120 ms), absence of Q waves in leads I and V6, delayed intrinsicoid deflection.",
        "treatment": "Typically no specific treatment is needed for LBBB itself, but the underlying cause (like heart disease) should be addressed.",
        "severity": "Moderate",
        "color": "#FFA500"  # Orange
    },
    "Normal": {
        "description": "Normal sinus rhythm indicates a healthy heart electrical system with regular electrical impulses originating from the sinoatrial node.",
        "indicators": "Regular rhythm, normal P waves followed by QRS complexes, normal PR and QT intervals.",
        "treatment": "No treatment required.",
        "severity": "None",
        "color": "#4CAF50"  # Green
    },
    "Premature Atrial Contraction": {
        "description": "Premature Atrial Contraction (PAC) occurs when the atria contract early, causing an irregular heartbeat.",
        "indicators": "Early P wave with abnormal morphology, followed by a normal QRS complex, often with a compensatory pause.",
        "treatment": "Usually no treatment needed for occasional PACs. Lifestyle modifications may help reduce frequency.",
        "severity": "Low",
        "color": "#FFEB3B"  # Yellow
    },
    "Ventricular Fibrillation": {
        "description": "Ventricular Fibrillation (VF) is a life-threatening condition where the ventricles quiver instead of pumping blood effectively.",
        "indicators": "Chaotic, irregular waveforms without distinct P waves, QRS complexes, or T waves.",
        "treatment": "Immediate defibrillation and advanced cardiac life support (ACLS) are required.",
        "severity": "Critical",
        "color": "#F44336"  # Red
    },
    "Premature Ventricular Contraction": {
        "description": "Premature Ventricular Contraction (PVC) occurs when the ventricles contract early, disrupting the normal heart rhythm.",
        "indicators": "Wide and bizarre QRS complex, no preceding P wave, and often followed by a compensatory pause.",
        "treatment": "For occasional PVCs, no treatment may be needed. Frequent PVCs may require medication or ablation.",
        "severity": "Low to Moderate",
        "color": "#FF9800"  # Amber
    },
    "Right Bundle Branch Block": {
        "description": "Right Bundle Branch Block (RBBB) is a cardiac conduction abnormality where electrical impulses are delayed or blocked along the right bundle branch.",
        "indicators": "Wide QRS complex (≥120 ms), RSR' pattern in V1-V3, wide S waves in leads I and V6.",
        "treatment": "Typically no specific treatment is needed for RBBB itself, but the underlying cause should be addressed.",
        "severity": "Moderate",
        "color": "#FFC107"  # Amber
    }
}

# Process the uploaded image
if uploaded_file and model_loaded:
    with col2:
        # Display the uploaded image
        st.image(uploaded_file, caption="Uploaded ECG Image", use_column_width=True)
        
        # Add a spinner during processing
        with st.spinner("Analyzing ECG pattern..."):
            # Simulate processing time for better UX
            progress_bar = st.progress(0)
            for i in range(100):
                time.sleep(0.02)
                progress_bar.progress(i + 1)
            
            # Process the image
            img = image.load_img(uploaded_file, target_size=(64, 64))
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            
            # Make prediction
            pred = model.predict(x)
            y_pred = np.argmax(pred)
            
            # Map prediction to condition
            index = ['Left Bundle Branch Block', "Normal", "Premature Atrial Contraction", 
                    "Ventricular Fibrillation", 'Premature Ventricular Contraction', 
                    "Right Bundle Branch Block"]
            result = str(index[y_pred])
            
            # Get confidence scores
            confidence_scores = pred[0] * 100
            
    # Display results
    st.markdown("---")
    st.markdown("<h2 style='text-align: center;'>Analysis Results</h2>", unsafe_allow_html=True)
    
    col_res1, col_res2 = st.columns([1, 1])
    
    with col_res1:
        # Display the prediction with styling based on condition
        st.markdown(f"""
        <div class='result-text' style='background-color: {condition_info[result]["color"]}20; 
                                        color: {condition_info[result]["color"]}; 
                                        border: 2px solid {condition_info[result]["color"]};'>
            Predicted Condition: {result}
            <br>Confidence: {confidence_scores[y_pred]:.2f}%
            <br>Severity: {condition_info[result]["severity"]}
        </div>
        """, unsafe_allow_html=True)
        
        # Display information about the condition
        st.markdown("### Condition Information")
        st.markdown(f"**Description:** {condition_info[result]['description']}")
        st.markdown(f"**Indicators:** {condition_info[result]['indicators']}")
        st.markdown(f"**Treatment:** {condition_info[result]['treatment']}")
        
    with col_res2:
        # Create a bar chart of confidence scores
        fig, ax = plt.subplots(figsize=(10, 6))
        colors = [condition_info[cond]["color"] for cond in index]
        ax.barh(index, confidence_scores, color=[f"{c}80" for c in colors])
        ax.set_xlabel('Confidence (%)')
        ax.set_title('Confidence Scores for Each Condition')
        ax.grid(axis='x', linestyle='--', alpha=0.7)
        
        # Highlight the predicted condition
        highlight_index = index.index(result)
        ax.barh(index[highlight_index], confidence_scores[highlight_index], color=condition_info[result]["color"])
        
        st.pyplot(fig)
        
        # Add a disclaimer
        st.markdown("""
        **Disclaimer:** This analysis is for educational purposes only and should not replace 
        professional medical advice. Always consult with a healthcare provider for proper diagnosis 
        and treatment.
        """)

# Display a message if no file is uploaded
if not uploaded_file:
    with col2:
        st.markdown("""
        <div style="display: flex; justify-content: center; align-items: center; height: 300px; 
                    background-color: #f0f2f6; border-radius: 10px; border: 2px dashed #cccccc;">
            <div style="text-align: center;">
                <h3>Upload an ECG image to begin analysis</h3>
                <p>Supported format: PNG</p>
            </div>
        </div>
        """, unsafe_allow_html=True)

# Footer
st.markdown("""
<div class='footer'>
    <p>ECG Analyzer v1.0 | Powered by TensorFlow and Streamlit</p>
    <p>© 2025 ECG Analysis Project</p>
</div>
""", unsafe_allow_html=True)
