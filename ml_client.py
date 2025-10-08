#!/usr/bin/env python3
"""
Simple client example using functions from ml_service.py
No classes needed - just import and use the functions directly!
"""

# Import the client functions from the service file
from ml_service import (
    check_service_health,
    get_model_info,
    classify_dicom_file,
    classify_dicom_simple
)
import json
import os

# Example of how to use in application
def ml_classifier_example(dicom_file):
    """Example of how to use these functions in your application"""
    print("Checking service health...")
    health = check_service_health()
    if health:
        print(f"   ✓ Service is healthy: {health['status']}")
        print(f"   ✓ Device: {health['device']}")
    else:
        print("   ✗ Service is not available")
        return
    # Simple usage - just get the results
    hair, skin = classify_dicom_simple(dicom_file)
    print(f"Patient has {hair} hair and {skin} skin type")
    
    # Advanced usage - get full results with confidence scores
    result = classify_dicom_file(dicom_file)
    if result:
        hair_scores = result['results']['scores']['hair']
        skin_scores = result['results']['scores']['skin']
        print(f"Hair confidence: {hair_scores}")
        print(f"Skin confidence: {skin_scores}")

if __name__ == "__main__":
    dicom_file = "2024-03-18-14-39-21_d5b95953-c78a-4cc4-b596-c65caa435990.dcm"
    ml_classifier_example(dicom_file)
