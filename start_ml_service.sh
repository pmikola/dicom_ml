#!/bin/bash
# Simple wrapper script to start ML service with proper venv activation
# Usage: ./start_ml_service.sh

cd ~/hairskin_classifier
exec ./venv/bin/python ml_service.py
