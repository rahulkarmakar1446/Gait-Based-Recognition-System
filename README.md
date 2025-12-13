# Gait-Based-Recognition-System
“A Gait-Based Recognition System Invariant to Clothes.”
# 🚶‍♂️ Gait-Based Recognition System (Invariant to Clothes)

This repository implements a **Gait-Based Human Recognition System** that identifies individuals based on their walking patterns (gait), while being **robust to clothing variations**.  
The system leverages **Gait Energy Images (GEI)** and machine learning / deep learning techniques to extract discriminative gait features and perform recognition.

---

## 🔧 Features

- Gait-based biometric recognition independent of clothing changes
- Silhouette processing and **Gait Energy Image (GEI)** generation
- Feature extraction using deep learning / CNN-based models
- Model training, testing, and evaluation pipeline
- Modular and clean project structure
- Jupyter Notebook for experimentation and visualization
- Easy extension for real-time or video-based gait recognition

---

## 📁 Repository Structure

Gait-Based-Recognition-System/
├── README.md
├── LICENSE
├── .gitignore
├── notebooks/
│ └── gait_recognition.ipynb # EDA, preprocessing, training & evaluation
├── src/
│ ├── preprocessing.py # Silhouette processing & GEI generation
│ ├── feature_extraction.py # CNN / feature extraction logic
│ ├── model.py # Model architecture & training functions
│ ├── inference.py # Prediction / testing pipeline
│ └── utils.py # Helper functions & metrics
├── models/ # Saved trained models (.h5, .pkl)
├── results/ # Evaluation results, plots, confusion matrix
├── requirements.txt # Python dependencies
└── scripts/
├── train.py # Training script
└── test.py # Testing / inference script


---

## ⚙️ Setup Instructions

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/rahulkarmakar1446/Gait-Based-Recognition-System.git
cd Gait-Based-Recognition-System

2️⃣ Install Dependencies
pip install -r requirements.txt

3️⃣ Run the Jupyter Notebook

The main experimentation and training workflow is provided in a notebook.

jupyter notebook notebooks/gait_recognition.ipynb


This notebook includes:

Data preprocessing

GEI generation

Model training

Evaluation and visualization

4️⃣ Train the Model (Script-based)
python scripts/train.py


Trained models will be saved in the models/ directory.

5️⃣ Test / Run Inference
python scripts/test.py --input_path path_to_gei_or_image

🧪 Usage Example
from src.model import load_model
from src.inference import run_inference
from src.preprocessing import compute_gei

model = load_model("models/trained_model.h5")
gei = compute_gei("sample_input.png")

prediction = run_inference(model, gei)
print("Predicted Identity:", prediction)

## 📊 Performance Improvement & Accuracy Gain

This project demonstrates a **significant improvement in gait recognition accuracy** by applying robust preprocessing, Gait Energy Image (GEI) representation, and optimized feature extraction techniques.

### 🔹 Accuracy Improvement
- **Baseline Accuracy:** 79%
- **Final Model Accuracy:** **93.86%**
- **Overall Accuracy Gain:** **+14.86%**

This improvement highlights the effectiveness of:
- Clothing-invariant gait representations (GEI)
- Enhanced silhouette preprocessing
- Deep learning–based feature extraction

## 📈 Training Progress Visualization

The following figure illustrates the **training progress of the gait recognition model**, showing how performance improves across epochs.

- Accuracy increases steadily during training
- Loss decreases, indicating better model convergence
- Final model achieves **93.86% accuracy**, improving from an initial **79% baseline**

### 🔹 Training Accuracy & Loss Curve

![Training Progress](results/CNN Training Progress.png)


🔄 Customization & Extensions

Replace GEI with other gait representations

Experiment with different CNN architectures

Add temporal models (LSTM / GRU)

Extend to video-based gait recognition

Deploy as a REST API (Flask / FastAPI)

Integrate real-time webcam input

📚 References

Gait Energy Image (GEI) – Han & Bhanu

CASIA Gait Dataset

Deep Learning for Gait Recognition

Scikit-learn & TensorFlow documentation

👤 Author

Rahul Karmakar
GitHub: rahulkarmakar1446
