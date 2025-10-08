#!/usr/bin/env python3
"""
FastAPI service for HairSkinClassifier model on Jetson Nano.
Provides REST API endpoints for DICOM image classification.
Auto-activates virtual environment if needed.
"""

import os
import sys

# Auto-activate venv if not already activated
def ensure_venv_activated():
    """Automatically activate the virtual environment if not already active"""
    # Check if we're already in a venv
    if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print("Virtual environment already activated")
        return
    
    # Get the script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    venv_python = os.path.join(script_dir, 'venv', 'bin', 'python')
    
    # Check if venv exists
    if os.path.exists(venv_python):
        print("Restarting with venv Python: " + venv_python)
        # Re-execute this script with the venv Python
        os.execv(venv_python, [venv_python] + sys.argv)
    else:
        print("Warning: Virtual environment not found at " + venv_python)
        print("Continuing with system Python...")

# Ensure venv is activated before importing heavy libraries
ensure_venv_activated()

# Now we can safely import everything that requires Python 3.7+
import io
import base64
import logging
from typing import Dict, Any, Optional
import uvicorn
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse
import torch
import pydicom
import numpy as np
from PIL import Image
import requests

# Import the model
from DepiModels import HairSkinClassifier

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="HairSkin Classifier API",
    description="ML service for classifying hair color and skin type from DICOM images",
    version="1.0.0"
)

# Global model instance
model = None
device = None

@app.on_event("startup")
async def startup_event():
    """Initialize the model on startup"""
    global model, device
    
    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    try:
        # Initialize model
        model = HairSkinClassifier()
        logger.info("Model initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize model: {e}")
        raise

def load_dicom_dataset(dicom_data):
    """Load DICOM dataset for model input - matches main.py approach"""
    try:
        # Read DICOM file directly (just like main.py does)
        ds = pydicom.dcmread(io.BytesIO(dicom_data), force=True)
        
        logger.info(f"DICOM loaded - shape: {ds.pixel_array.shape}, dtype: {ds.pixel_array.dtype}")
        
        return ds
    except Exception as e:
        logger.error(f"Error loading DICOM: {e}")
        raise HTTPException(status_code=400, detail=f"Invalid DICOM file: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "device": str(device),
        "model_loaded": model is not None,
        "cuda_available": torch.cuda.is_available()
    }

@app.get("/model/info")
async def model_info():
    """Get model information"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "model_type": "HairSkinClassifier",
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "model_components": {
            "hair_color_model": hasattr(model, 'hc_model'),
            "skin_type_model": hasattr(model, 'skin_type_model')
        }
    }

@app.post("/classify/dicom")
async def classify_dicom_file(file: UploadFile = File(...)):
    """Classify hair color and skin type from DICOM file"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Read uploaded file
        dicom_data = await file.read()
        
        # Load DICOM dataset (just like main.py does)
        ds = load_dicom_dataset(dicom_data)
        
        # Run classification - pass DICOM dataset directly to model (matches main.py)
        with torch.no_grad():
            raw_output, colors = model(ds)
            hair_logits, skin_logits = raw_output
            hair_color, skin_color = colors
            
            # Print debug info
            logger.info(f"Device: {device}")
            logger.info(f"Raw hair logits: {hair_logits}")
            logger.info(f"Raw skin logits: {skin_logits}")
            logger.info(f"Hair color: {hair_color}")
            logger.info(f"Skin color: {skin_color}")
        
        return {
            "status": "success",
            "results": {
                "hair_color": hair_color,
                "skin_color": skin_color,
                "scores": {
                    "hair": hair_logits.tolist(),
                    "skin": skin_logits.tolist()
                }
            },
            "metadata": {
                "filename": file.filename,
                "device": str(device),
                "dicom_shape": str(ds.pixel_array.shape)
            }
        }
        
    except Exception as e:
        logger.error(f"Classification error: {e}")
        raise HTTPException(status_code=500, detail=f"Classification failed: {str(e)}")

# Client functions for easy integration
def check_service_health(base_url="http://localhost:8000"):
    """Check if the ML service is healthy"""
    try:
        response = requests.get(base_url + "/health", timeout=5)
        return response.json() if response.status_code == 200 else None
    except:
        return None

def get_model_info(base_url="http://localhost:8000"):
    """Get model information"""
    try:
        response = requests.get(base_url + "/model/info", timeout=5)
        return response.json() if response.status_code == 200 else None
    except:
        return None

def classify_dicom_file(dicom_path, base_url="http://localhost:8000"):
    """Classify a DICOM file using the ML service"""
    try:
        with open(dicom_path, 'rb') as f:
            files = {'file': (os.path.basename(dicom_path), f, 'application/dicom')}
            response = requests.post(base_url + "/classify/dicom", files=files, timeout=30)
            return response.json() if response.status_code == 200 else None
    except:
        return None

def classify_dicom_simple(dicom_path, base_url="http://localhost:8000"):
    """Simple classification that returns just hair and skin type"""
    result = classify_dicom_file(dicom_path, base_url)
    if result and result.get('status') == 'success':
        results = result['results']
        return results['hair_color'], results['skin_color']
    return None, None

if __name__ == "__main__":
    print("Starting HairSkin Classifier ML Service...")
    print("Python executable: " + sys.executable)
    print("Working directory: " + os.getcwd())
    
    # Start the FastAPI server
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000,
        log_level="info"
    )