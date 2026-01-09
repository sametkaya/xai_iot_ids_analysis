# XAI-IoT-IDS: Explainable AI for IoT Intrusion Detection Systems

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.12%2B-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A comprehensive framework for evaluating machine learning and deep learning models on IoT network intrusion detection with explainable AI (XAI) techniques.

## 📋 Overview

This repository contains the implementation code for our research paper:

> **"Comparative Analysis of Machine Learning and Deep Learning Models for IoT Intrusion Detection: A Multi-Dataset Evaluation with Explainable AI"**

The framework provides:
- **15 Classification Models**: Traditional ML, Ensemble Methods, and Deep Learning
- **4 IoT Datasets**: TON-IoT, UNSW-NB15, CICIoT2023, Edge-IIoT
- **XAI Integration**: SHAP and LIME explanations for model interpretability
- **Automated Hyperparameter Optimization**: Using Optuna with TPE sampler
- **GPU Acceleration**: Support for CUDA-enabled training

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    XAI-IoT-IDS Framework                        │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐              │
│  │   TON-IoT   │  │  EDGE-IIot  │  │  CICIoT2023 │              │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘              │
│         └────────────────┼────────────────┘                     │
│                          ▼                                      │
│              ┌───────────────────────┐                          │
│              │   Data Preprocessing  │                          │
│              │  • Label Encoding     │                          │
│              │  • Feature Scaling    │                          │
│              │  • SMOTE Balancing    │                          │
│              └───────────┬───────────┘                          │
│                          ▼                                      │
│    ┌─────────────────────────────────────────────────┐          │
│    │              Model Training                     │          │
│    │  ┌─────────┐ ┌─────────┐ ┌─────────┐            │          │
│    │  │   ML    │ │Ensemble │ │   DL    │            │          │
│    │  │  Models │ │ Models  │ │ Models  │            │          │
│    │  └─────────┘ └─────────┘ └─────────┘            │          │
│    │       + Optuna Hyperparameter Optimization      │          │
│    └─────────────────────┬───────────────────────────┘          │
│                          ▼                                      │
│              ┌───────────────────────┐                          │
│              │   XAI Explanations    │                          │
│              │  • SHAP TreeExplainer │                          │
│              │  • LIME Tabular       │                          │
│              └───────────────────────┘                          │
└─────────────────────────────────────────────────────────────────┘
```

## 📊 Supported Models

### Traditional Machine Learning
| Model | Description |
|-------|-------------|
| Logistic Regression | Linear baseline classifier |
| Decision Tree | Single tree classifier |
| K-Nearest Neighbors | Distance-based classifier |
| Naive Bayes | Probabilistic classifier |

### Ensemble Methods
| Model | Description |
|-------|-------------|
| Random Forest | Bagging ensemble of decision trees |
| XGBoost | Gradient boosting with regularization |
| LightGBM | Histogram-based gradient boosting |
| CatBoost | Gradient boosting with categorical features |

### Deep Learning
| Model | Description |
|-------|-------------|
| MLP | Multi-Layer Perceptron |
| CNN-1D | 1D Convolutional Neural Network |
| LSTM | Long Short-Term Memory |
| GRU | Gated Recurrent Unit |
| Transformer | Self-attention based model |

## 📁 Supported Datasets

| Dataset | Classes | Features | Samples | Domain |
|---------|---------|----------|---------|--------|
| [TON-IoT](https://research.unsw.edu.au/projects/toniot-datasets) | 10 | 44 | 461,043 | IoT/IIoT |
| [UNSW-NB15](https://research.unsw.edu.au/projects/unsw-nb15-dataset) | 10 | 49 | 257,673 | Network |
| [CICIoT2023](https://www.unb.ca/cic/datasets/iotdataset-2023.html) | 8 | 46 | 33M+ | IoT |
| [Edge-IIoT](https://www.kaggle.com/datasets/mohamedamineferrag/edgeiiotset-cyber-security-dataset-of-iot-iiot) | 15 | 61 | 2M+ | Edge/IIoT |

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- CUDA 11.6+ (optional, for GPU acceleration)

### Setup

1. **Clone the repository**
```bash
git clone https://github.com/sametkaya/xai-iot-ids.git
cd xai-iot-ids
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Download datasets**

Place your datasets in the `dataset/` directory:
```
dataset/
├── toniot/
│   └── *.csv
├── unsw-nb15/
│   └── *.csv
├── ciciot2023/
│   └── *.csv
└── edge-iiot/
    └── *.csv
```

## 💻 Usage

### Basic Usage

1. **Configure the script**

Edit the global variables in `xai_iot_ids_run.py`:
```python
# Data directory
DATA_DIR = 'dataset/toniot'

# Output directory
OUTPUT_DIR = 'outputs/toniot'

# Select models to train
ENABLED_ML_MODELS = ['XGBoost', 'LightGBM', 'Random_Forest', 'CatBoost']
ENABLED_BASELINE_MODELS = ['Logistic_Regression', 'Decision_Tree', 'KNN', 'Naive_Bayes']
ENABLED_DL_MODELS = ['MLP', 'CNN_1D', 'LSTM', 'GRU', 'Transformer']
```

2. **Run the training**
```bash
python xai_iot_ids_run.py
```

### Configuration Options

```python
CONFIG = {
    'subsample_size': 2_000_000,    # Max samples to use
    'test_size': 0.2,               # Test set ratio
    'val_size': 0.1,                # Validation set ratio
    'random_state': 42,             # Random seed
    'n_features': 35,               # Number of features to select
    
    # Optuna hyperparameter optimization
    'optuna_trials_ml': 50,         # Trials for ML models
    'optuna_trials_dl': 30,         # Trials for DL models
    
    # Deep Learning
    'epochs': 100,                  # Max training epochs
    'batch_size': 2048,             # Batch size
    'early_stopping_patience': 15,  # Early stopping patience
    
    # Explainability
    'shap_samples': 500,            # Samples for SHAP analysis
    'lime_samples': 5,              # Samples for LIME analysis
}
```

## 📈 Output Structure

After running, results are saved in the output directory:

```
outputs/toniot/
├── toniot_overall_summary.json      # Summary of all models
├── XGBoost/
│   ├── toniot_XGBoost_results.json       # Performance metrics
│   ├── toniot_XGBoost_best_params.json   # Optimized hyperparameters
│   ├── toniot_XGBoost_model.joblib       # Trained model
│   ├── toniot_XGBoost_confusion_matrix.png
│   ├── toniot_XGBoost_shap_importance.png
│   ├── toniot_XGBoost_shap_importance.csv
│   ├── toniot_XGBoost_lime_explanations.png
│   └── toniot_XGBoost_experiment_summary.txt
├── LightGBM/
│   └── ...
└── ...
```

## 📊 Evaluation Metrics

The framework computes comprehensive metrics:

- **Accuracy**: Overall correct predictions
- **Precision (Macro)**: Average precision across classes
- **Recall (Macro)**: Average recall across classes
- **F1-Score (Macro/Weighted)**: Harmonic mean of precision and recall
- **MCC**: Matthews Correlation Coefficient
- **Balanced Accuracy**: Average recall per class
- **ROC-AUC**: Area under ROC curve (where applicable)

## 🔍 Explainability Features

### SHAP (SHapley Additive exPlanations)
- TreeExplainer for tree-based models
- Feature importance rankings
- Summary plots and dependence plots

### LIME (Local Interpretable Model-agnostic Explanations)
- Instance-level explanations
- Feature contribution analysis
- Support for all model types

## 🖥️ Hardware Requirements

### Minimum
- CPU: 4 cores
- RAM: 16 GB
- Storage: 50 GB

### Recommended (for full experiments)
- CPU: 8+ cores
- RAM: 32+ GB
- GPU: NVIDIA with 8+ GB VRAM (CUDA 11.6+)
- Storage: 100+ GB SSD

## 📝 Citation

If you use this code in your research, please cite:

```bibtex
@article{kaya2025xai_iot_ids,
  title={Comparative Analysis of Machine Learning and Deep Learning Models for IoT Intrusion Detection: A Multi-Dataset Evaluation with Explainable AI},
  author={Kaya, Samet and [Co-authors]},
  journal={EURASIP Journal on Information Security},
  year={2025},
  note={Under Review}
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📧 Contact

For questions or feedback, please open an issue or contact:
- **Author**: Samet Kaya
- **Email**: [your-email@example.com]

## 🙏 Acknowledgments

- Dataset providers: UNSW Sydney, UNB CIC, Kaggle
- Open-source libraries: scikit-learn, PyTorch, XGBoost, LightGBM, CatBoost, SHAP, LIME, Optuna
