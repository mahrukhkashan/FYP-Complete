#!/usr/bin/env python3
"""
Main script to run the Sepsis Prediction System
"""

import os
import sys
import argparse
from api.app import app
from train_model import train_and_save_model




def main():
    parser = argparse.ArgumentParser(description='Sepsis Prediction System')
    parser.add_argument('--train', action='store_true', help='Train the model before starting')
    parser.add_argument('--port', type=int, default=5000, help='Port to run the server on')
    parser.add_argument('--host', default='0.0.0.0', help='Host to run the server on')
    parser.add_argument('--debug', action='store_true', help='Run in debug mode')
    
    args = parser.parse_args()
    
    # Create necessary directories
    os.makedirs('models/saved_models', exist_ok=True)
    os.makedirs('uploads', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    
    # Train model if requested
    if args.train:
        print("Training model...")
        train_and_save_model()
        print("Model training completed!")
    
    # Start the Flask app
    print(f"Starting Sepsis Prediction System on http://{args.host}:{args.port}")
    app.run(
    host=args.host,
    port=args.port,
    debug=args.debug,
    use_reloader=False
)


if __name__ == '__main__':
    main()