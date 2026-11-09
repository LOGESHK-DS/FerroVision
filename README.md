# F E R R O   V I S I O N
### Intelligent Predictive Maintenance & Automated Quality Inspection

**An end-to-end Deep Learning system combining Remaining Useful Life (RUL) forecasting and automated industrial defect detection.**

<p align="center">
  <strong>Predictive Maintenance · Time-Series Forecasting · Computer Vision · Industrial AI</strong>
</p>

---

## 1. Overview

FerroVision is an industrial intelligence project designed to address two important challenges in manufacturing and asset-intensive industries: **predicting equipment degradation before failure and identifying defective manufactured components through computer vision.**

The project combines deep learning models for predictive maintenance with image classification models for automated quality inspection.

It uses two different industrial datasets:

- **Turbofan Engine Degradation Dataset:** Multivariate sensor readings collected across operating cycles, used to predict Remaining Useful Life (RUL).
- **Casting Product Image Dataset:** Industrial casting images categorized as normal or defective, used to develop automated visual inspection models.

The trained models are integrated into an interactive Streamlit application where users can explore sensor data, generate RUL forecasts, inspect casting images and compare model performance.

The objective is to demonstrate how machine learning can move industrial operations from reactive maintenance and manual inspection towards data-driven decision-making.

---

## 2. Problem We Are Solving

Industrial equipment failures can result in unexpected downtime, expensive maintenance, production losses and operational risks.

Similarly, conventional manufacturing quality inspection often relies on manual visual examination, which can be time-consuming and inconsistent.

FerroVision addresses these challenges through three machine learning tasks:

| Task | Problem | Solution |
|---|---|---|
| RUL Regression | Estimating how many operating cycles remain before engine failure | Artificial Neural Network (ANN) |
| Time-Series RUL Forecasting | Learning degradation patterns from historical sensor sequences | RNN and LSTM |
| Visual Defect Classification | Identifying defective manufactured casting components | Custom CNN and transfer learning |

---

## 3. Datasets

### A. Turbofan Engine Degradation Dataset

Used for both regression and time-series forecasting.

| Attribute | Description |
|---|---|
| Records | Approximately 61,000+ |
| Features | 27 columns |
| Engine identifier | engine_id |
| Operating index | cycle |
| Operational settings | 3 |
| Sensor readings | 21 |
| Target | Remaining Useful Life (RUL) |

The dataset contains engine operating conditions and sensor readings over multiple operating cycles.

**Objective:** Predict the number of remaining operating cycles before an engine reaches failure.

### B. Casting Product Image Dataset

Used for binary image classification.

| Attribute | Description |
|---|---|
| Total images | Approximately 7,348 |
| Image dimensions | 300 × 300 pixels |
| Classes | Normal and Defective |
| Training images | 6,633 |
| Testing images | 715 |

Class distribution:

| Dataset | Normal | Defective |
|---|---:|---:|
| Training | 2,875 | 3,758 |
| Testing | 262 | 453 |

**Objective:** Automatically distinguish defective casting components from normal manufactured components.

---

## 4. Technical Implementation

### Task 1: Remaining Useful Life Regression

**Model: Artificial Neural Network (ANN)**

The ANN learns the relationship between individual operating-cycle sensor measurements and the corresponding RUL.

Implementation:
- Feature selection and preprocessing.
- StandardScaler-based feature scaling.
- Dense neural network architecture.
- ReLU activation and dropout regularization.
- Regression using Mean Squared Error loss.
- Evaluation using MAE, RMSE and R².

### Task 2: Time-Series RUL Forecasting

**Models: Simple Recurrent Neural Network (RNN) and Long Short-Term Memory (LSTM)**

Unlike the ANN, which uses individual operating-cycle measurements, the sequential models learn from historical sensor observations.

Implementation:
- Grouping observations by engine.
- Fixed-length sliding-window sequence generation.
- Sequential feature learning.
- SimpleRNN and LSTM architectures.
- Early stopping based on validation loss.
- Model checkpointing to preserve the best validation-selected model.

The objective is to evaluate whether temporal information improves RUL prediction compared with the ANN baseline.

### Task 3: Automated Casting Defect Classification

Three image classification architectures are used:

1. Custom Convolutional Neural Network (CNN).
2. MobileNetV2.
3. EfficientNetB0.

Implementation:
- Image preprocessing and resizing.
- Binary classification of Normal and Defective components.
- Custom CNN architecture.
- Transfer learning using pretrained CNN architectures.
- Model evaluation using Accuracy, Precision, Recall and F1-score.
- Confusion-matrix-based evaluation.

MobileNetV2 uses a validation-calibrated classification threshold of 0.31 in the deployed application, while the other classifiers use a threshold of 0.50.

---

## 5. Model Evaluation

### RUL Model Comparison

The regression models are evaluated using:

| Metric | Interpretation |
|---|---|
| MAE | Average absolute prediction error |
| RMSE | Penalizes larger prediction errors |
| R² | Measures the proportion of target variance explained by the model |

The three models are compared using held-out engine data.

### Casting Classification Comparison

The image classification models are evaluated using:

| Metric | Interpretation |
|---|---|
| Accuracy | Overall classification correctness |
| Precision | Reliability of positive defect predictions |
| Recall | Ability to identify actual defective components |
| F1-score | Balance between precision and recall |

Confusion matrices provide additional insight into correct and incorrect classifications.

*Actual evaluation results are available in the project's outputs/metrics directory.*

---

## 6. What Has Been Improved During Development

The project was developed with particular attention to evaluation reliability, preprocessing consistency and deployment.

### 1. Engine-Level Data Splitting

Instead of randomly splitting individual sensor records, the RUL workflow splits data by engine identifiers.

This prevents observations from the same engine appearing across training, validation and testing partitions.

It provides a more meaningful evaluation of model generalization to unseen engines.

### 2. Leakage-Aware Feature Scaling

StandardScaler is fitted exclusively on training data.

The same fitted scaler is then applied to validation and test datasets.

This avoids allowing information from evaluation data to influence training preprocessing.

### 3. Validation-Based Model Selection

Early stopping monitors validation loss and restores the best-performing weights.

ModelCheckpoint preserves the best validation-selected model rather than simply saving the final training epoch.

### 4. Consistent Inference Preprocessing

The RUL workflow saves feature-column information and fitted scalers alongside the trained models.

This helps preserve feature ordering and preprocessing consistency during inference.

### 5. Threshold-Calibrated Defect Classification

MobileNetV2 uses a validation-calibrated threshold of 0.31 rather than automatically applying the conventional 0.50 threshold.

This allows classification decisions to reflect the selected operating threshold.

### 6. Interactive Model Deployment

Instead of limiting the work to offline model training, the trained models are integrated into a Streamlit application.

Users can interact with the models through a graphical interface without manually executing the training scripts.

---

## 7. FerroVision Application

The project includes an interactive Streamlit dashboard with four navigation sections.

| Section | Functionality |
|---|---|
| Overview | Explore turbofan data, dataset statistics and sensor readings |
| RUL Forecast | Select an ANN, RNN or LSTM model and generate an RUL prediction |
| Defect Classification | Upload a casting image and obtain a Normal or Defective prediction |
| Model Comparison | Compare saved RUL and casting classification evaluation metrics |

The application uses saved model artifacts and preprocessing information for inference.

---

## 8. Technology Stack

| Category | Technologies |
|---|---|
| Programming | Python |
| Data Processing | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn, Streamlit charts |
| Machine Learning | Scikit-learn |
| Deep Learning | TensorFlow, Keras |
| Computer Vision | CNN, MobileNetV2, EfficientNetB0 |
| Time-Series Modeling | RNN, LSTM |
| Application Development | Streamlit |
| Development Environment | VS Code, Python virtual environment |
| Version Control | Git, GitHub |

---

## 9. Project Structure

```text
FerroVision/
│
├── data/
│   ├── raw/
│   │   ├── train_data.csv
│   │   ├── test_data.csv
│   │   └── casting_data/
│   │       ├── train/
│   │       └── test/
│   │
│   └── processed/
│
├── models/
│   ├── rul/
│   │   ├── best_ann_final.keras
│   │   ├── best_rnn_final.keras
│   │   ├── best_lstm_final.keras
│   │   ├── feature_scalers
│   │   └── feature_columns
│   │
│   └── vision/
│       ├── Custom CNN
│       ├── MobileNetV2
│       └── EfficientNetB0
│
├── outputs/
│   └── metrics/
│       ├── ann_final_metrics.csv
│       ├── rnn_final_metrics.csv
│       ├── lstm_final_metrics.csv
│       └── casting model metrics
│
├── logs/
│   └── FerroVision.pdf
│
├── app.py
├── train_rul.py
├── train_vision.py
├── requirements.txt
└── README.md
```

*The structure above summarizes the principal project components. Exact artifact filenames may differ depending on the saved model versions.*

---

## 10. Installation and Execution

### Prerequisites

- Python 3.10 or later.
- pip.
- Git.

### Step 1: Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd FerroVision
```

### Step 2: Create a virtual environment

```bash
python -m venv ferrovision_env
```

Activate it:

Windows:

```bash
ferrovision_env\Scripts\activate
```

Linux/macOS:

```bash
source ferrovision_env/bin/activate
```

### Step 3: Install dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Launch FerroVision

```bash
streamlit run app.py
```

Open the local URL displayed in your terminal to access the application.

---

## 11. Practical Applications

FerroVision demonstrates potential applications across several industrial domains:

- **Predictive Maintenance:** Estimate equipment degradation and support maintenance planning.
- **Asset Reliability:** Monitor equipment health using operational sensor data.
- **Manufacturing Quality Control:** Identify defective casting components through automated image inspection.
- **Operational Risk Reduction:** Support early identification of assets approaching failure.
- **Industrial Analytics:** Bring multiple deep learning workflows into a single interactive application.

These models provide decision-support predictions and should not independently authorize maintenance or production-quality decisions without appropriate engineering validation.

---

## 12. Future Improvements

Potential directions for further development include:

- Advanced sensor degradation visualization and explainability.
- More extensive hyperparameter optimization.
- Prediction uncertainty estimation for RUL forecasting.
- Grad-CAM-based visual explanations for defect classification.
- Real-time sensor data integration.
- Model monitoring and drift detection.
- Deployment through a cloud-hosted application.
- Integration with industrial IoT and maintenance management systems.

---

## 13. Conclusion

FerroVision brings together predictive maintenance, temporal deep learning and computer vision within a unified industrial analytics application.

By developing ANN, RNN and LSTM models for RUL forecasting alongside CNN-based casting defect classifiers, the project explores how deep learning can support both equipment reliability and manufacturing quality inspection.

Beyond model development, the project emphasizes data leakage prevention, validation-based model selection, consistent inference preprocessing, model evaluation and interactive deployment.

**FerroVision: From equipment degradation to intelligent industrial inspection.**

---

### Disclaimer

FerroVision is an academic and technical demonstration of deep learning for industrial reliability and quality inspection. Its predictions are intended for decision support and require further validation before deployment in safety-critical industrial environments.