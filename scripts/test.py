# scripts/test.py

import argparse
import os
from src.model import load_model  # function to load your model
from src.preprocessing import compute_gei  # function to prepare input
from src.inference import run_inference  # function to run prediction

def main():
    parser = argparse.ArgumentParser(description="Test Gait Recognition Model")
    parser.add_argument('--model_path', type=str, default='models/trained_model.h5',
                        help='Path to the trained model file')
    parser.add_argument('--input_path', type=str, required=True,
                        help='Path to input silhouette or GEI image')
    args = parser.parse_args()

    # 1️⃣ Check if model exists
    if not os.path.exists(args.model_path):
        print(f"Model file not found at {args.model_path}. Please train the model first.")
        return

    # 2️⃣ Load model
    model = load_model(args.model_path)

    # 3️⃣ Preprocess input image
    input_gei = compute_gei(args.input_path)

    # 4️⃣ Run inference
    prediction = run_inference(model, input_gei)

    # 5️⃣ Print result
    print(f"Predicted ID / Label: {prediction}")

if __name__ == "__main__":
    main()

