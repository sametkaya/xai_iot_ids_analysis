"""
================================================================================
MULTI-DATASET MODEL COMPARISON WITH EXPLAINABILITY - V10 WSL/Ubuntu
================================================================================
V10 Updates:
1. Fixed: XGBoost SHAP "base_score" bug (multi-class array parsing error)
2. Added: Multiple fallback methods for XGBoost SHAP:
   - Method 1: Direct TreeExplainer
   - Method 2: shap.Explainer with tree algorithm
   - Method 3: Permutation explainer as last resort

V9 Updates:
1. Fixed: SHAP errors now printed to console with full traceback
2. Fixed: More robust multi-class SHAP value handling (3D arrays)
3. Fixed: SHAP/LIME success/failure messages in console output
4. Added: Shape debugging info for SHAP values

V8 Updates:
1. Fixed: Consistent report format across all models
2. Fixed: SHAP/LIME errors now shown in reports (not silently ignored)
3. All models now have sections 6 (SHAP) and 7 (LIME) with status info

V7 Updates:
1. Fixed GPU detection - uses simple test data instead of training data
2. XGBoost, LightGBM, CatBoost GPU enabled (stable on Linux)
3. CatBoost SHAP enabled (stable on Linux, skipped on Windows)
4. All 15 models active
5. Automatic dataset detection (4 datasets)
6. Smart early stopping & mode collapse detection

Supported Datasets:
- TON-IoT
- UNSW-NB15
- CICIoT2023
- Edge-IIoT

Usage:
    1. Change DATA_DIR and OUTPUT_DIR
    2. Run: python3 multi_dataset_comparison_v10.py
================================================================================
"""

# ============================================================================
# ðŸ”§ GLOBAL INPUTS - BUNLARI DEÄžÄ°ÅžTÄ°R!
# ============================================================================

# Veri seti dizini
DATA_DIR = 'dataset/toniot'

# Ã‡Ä±ktÄ± dizini
OUTPUT_DIR = 'outputs/toniot'

# Dataset adÄ± (dosya isimlendirmesi iÃ§in) - otomatik algÄ±lanacak ama manuel de ayarlanabilir
DATASET_NAME = None  # None ise otomatik algÄ±lanÄ±r: 'toniot', 'ciciot2023', 'unsw-nb15', 'edge-iiot'

# ============================================================================

# ============================================================================
# ðŸŽ¯ MODEL SEÃ‡Ä°MÄ° - Ä°STEDÄ°ÄžÄ°NÄ° COMMENT/UNCOMMENT YAP
# ============================================================================

# Optuna ile optimize edilecek ML modeller
ENABLED_ML_MODELS = [
    # 'XGBoost',
    # 'LightGBM',
    # 'Random_Forest',
    # 'CatBoost',

]

# Baseline ML modeller (optimizasyon yok)
ENABLED_BASELINE_MODELS = [
    # 'Logistic_Regression',
    # 'Decision_Tree',
    # 'KNN',
    # 'Naive_Bayes',
]

# Deep Learning modeller
ENABLED_DL_MODELS = [
    'MLP',
    # 'CNN_1D',
    # 'LSTM',
    # 'GRU',
    # 'Transformer',
]

import os
import sys
import json
import time
import warnings
import traceback
from datetime import datetime
from pathlib import Path

warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, RobustScaler, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    classification_report, f1_score, accuracy_score,
    precision_score, recall_score, balanced_accuracy_score,
    matthews_corrcoef, confusion_matrix, roc_auc_score
)

import optuna
from optuna.samplers import TPESampler
optuna.logging.set_verbosity(optuna.logging.WARNING)

from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from tqdm import tqdm
import joblib  # Model kaydetme iÃ§in

# SHAP & LIME
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("âš  SHAP not available. Install with: pip install shap")

try:
    import lime
    import lime.lime_tabular
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False
    print("âš  LIME not available. Install with: pip install lime")

# PyTorch
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts
    TORCH_AVAILABLE = True
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"âœ“ PyTorch: {DEVICE}")
except ImportError:
    TORCH_AVAILABLE = False
    DEVICE = None

try:
    from lightgbm import LGBMClassifier
    LGBM_AVAILABLE = True
except ImportError:
    LGBM_AVAILABLE = False

try:
    from catboost import CatBoostClassifier
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    print("âš  CatBoost not available. Install with: pip install catboost")


# ============================================================================
# CONFIG
# ============================================================================
CONFIG = {
    'subsample_size': 2_000_000,
    'test_size': 0.2,
    'val_size': 0.1,
    'random_state': 42,
    'n_features': 35,

    # Optuna
    'optuna_trials_ml': 50,   # ML modeller iÃ§in trial sayÄ±sÄ±
    'optuna_trials_dl': 30,   # DL modeller iÃ§in trial sayÄ±sÄ±

    # Deep Learning
    'epochs': 100,
    'batch_size': 2048,
    'early_stopping_patience': 15,

    # Explainability
    'shap_samples': 500,
    'lime_samples': 5,

    # Mode Collapse Detection
    'mode_collapse_threshold': 0.10,  # F1 < 0.10 ise mode collapse
    'max_retry_attempts': 3,          # KaÃ§ kez tekrar denenecek
}



# ============================================================================
# DATASET TAXONOMY MAPPING
# ============================================================================
TAXONOMY_MAPS = {
    'ciciot2023': {
        'label_col': 'Label',
        'alt_label_cols': ['label', 'class', 'attack_type'],
        'taxonomy': {
            'BENIGN': 'Benign',
            'DDOS-ICMP_FLOOD': 'DDoS', 'DDOS-UDP_FLOOD': 'DDoS', 'DDOS-TCP_FLOOD': 'DDoS',
            'DDOS-PSHACK_FLOOD': 'DDoS', 'DDOS-SYN_FLOOD': 'DDoS', 'DDOS-RSTFINFLOOD': 'DDoS',
            'DDOS-SYNONYMOUSIP_FLOOD': 'DDoS', 'DDOS-ICMP_FRAGMENTATION': 'DDoS',
            'DDOS-UDP_FRAGMENTATION': 'DDoS', 'DDOS-ACK_FRAGMENTATION': 'DDoS',
            'DDOS-HTTP_FLOOD': 'DDoS', 'DDOS-SLOWLORIS': 'DDoS',
            'DOS-UDP_FLOOD': 'DoS', 'DOS-TCP_FLOOD': 'DoS', 'DOS-SYN_FLOOD': 'DoS', 'DOS-HTTP_FLOOD': 'DoS',
            'MIRAI-GREETH_FLOOD': 'Mirai', 'MIRAI-UDPPLAIN': 'Mirai', 'MIRAI-GREIP_FLOOD': 'Mirai',
            'VULNERABILITYSCAN': 'Recon', 'RECON-HOSTDISCOVERY': 'Recon', 'RECON-OSSCAN': 'Recon',
            'RECON-PORTSCAN': 'Recon', 'RECON-PINGSWEEP': 'Recon',
            'MITM-ARPSPOOFING': 'Spoofing', 'DNS_SPOOFING': 'Spoofing',
            'DICTIONARYBRUTEFORCE': 'BruteForce', 'BROWSERHIJACKING': 'BruteForce',
            'COMMANDINJECTION': 'Web', 'SQLINJECTION': 'Web', 'XSS': 'Web',
            'BACKDOOR_MALWARE': 'Web', 'UPLOADING_ATTACK': 'Web',
        },
        'drop_cols': [],
        'headers': None,  # Has headers
    },
    'unsw-nb15': {
        'label_col': 'attack_cat',
        'alt_label_cols': ['Attack_cat', 'attack_category', 'Label', 'label'],
        'taxonomy': {
            'Normal': 'Normal', 'Fuzzers': 'Fuzzers', 'Analysis': 'Analysis',
            'Backdoor': 'Backdoor', 'Backdoors': 'Backdoor', 'DoS': 'DoS',
            'Exploits': 'Exploits', 'Generic': 'Generic', 'Reconnaissance': 'Reconnaissance',
            'Shellcode': 'Shellcode', 'Worms': 'Worms',
            'normal': 'Normal', 'fuzzers': 'Fuzzers', 'analysis': 'Analysis',
            'backdoor': 'Backdoor', 'dos': 'DoS', 'exploits': 'Exploits',
            'generic': 'Generic', 'reconnaissance': 'Reconnaissance',
            'shellcode': 'Shellcode', 'worms': 'Worms',
            # Empty/NaN -> Normal (binary label=0)
            '': 'Normal', ' ': 'Normal',
        },
        'drop_cols': ['id', 'srcip', 'dstip', 'attack_cat', 'label', 'Label'],
        'headers': [
            'srcip', 'sport', 'dstip', 'dsport', 'proto', 'state', 'dur',
            'sbytes', 'dbytes', 'sttl', 'dttl', 'sloss', 'dloss', 'service',
            'Sload', 'Dload', 'Spkts', 'Dpkts', 'swin', 'dwin', 'stcpb',
            'dtcpb', 'smeansz', 'dmeansz', 'trans_depth', 'res_bdy_len',
            'Sjit', 'Djit', 'Stime', 'Ltime', 'Sintpkt', 'Dintpkt',
            'tcprtt', 'synack', 'ackdat', 'is_sm_ips_ports', 'ct_state_ttl',
            'ct_flw_http_mthd', 'is_ftp_login', 'ct_ftp_cmd', 'ct_srv_src',
            'ct_srv_dst', 'ct_dst_ltm', 'ct_src_ltm', 'ct_src_dport_ltm',
            'ct_dst_sport_ltm', 'ct_dst_src_ltm', 'attack_cat', 'Label'
        ],
    },
    'toniot': {
        'label_col': 'type',
        'alt_label_cols': ['Type', 'label', 'Label', 'attack_type'],
        'taxonomy': {
            'normal': 'Normal', 'backdoor': 'Backdoor', 'ddos': 'DDoS',
            'dos': 'DoS', 'injection': 'Injection', 'mitm': 'MITM',
            'password': 'Password', 'ransomware': 'Ransomware',
            'scanning': 'Scanning', 'xss': 'XSS',
            'Normal': 'Normal', 'Backdoor': 'Backdoor', 'DDoS': 'DDoS',
            'DoS': 'DoS', 'Injection': 'Injection', 'MITM': 'MITM',
            'Password': 'Password', 'Ransomware': 'Ransomware',
            'Scanning': 'Scanning', 'XSS': 'XSS',
        },
        'drop_cols': ['ts', 'src_ip', 'dst_ip', 'type', 'label'],
        'headers': None,  # Has headers
    },
    'edge-iiot': {
        'label_col': 'Attack_type',
        'alt_label_cols': ['attack_type', 'Attack_Type', 'label', 'Label'],
        'taxonomy': {
            'Normal': 'Normal', 'DDoS_UDP': 'DDoS', 'DDoS_ICMP': 'DDoS',
            'DDoS_HTTP': 'DDoS', 'DDoS_TCP': 'DDoS', 'SQL_injection': 'Injection',
            'Vulnerability_scanner': 'Reconnaissance', 'Password': 'BruteForce',
            'Uploading': 'Malware', 'Backdoor': 'Backdoor', 'Port_Scanning': 'Reconnaissance',
            'XSS': 'XSS', 'Ransomware': 'Ransomware', 'Fingerprinting': 'Reconnaissance',
            'MITM': 'MITM',
        },
        'drop_cols': ['Attack_type', 'Attack_label'],
        'headers': None,  # Has headers
    },
}


# ============================================================================
# HELPER: Get output filename with dataset and model prefix
# ============================================================================
def get_output_filename(model_name, filename, extension=None):
    """
    Generate output filename with dataset and model prefix.
    Format: {dataset}_{model}_{filename}.{ext}
    """
    global DATASET_NAME

    if extension:
        return f"{DATASET_NAME}_{model_name}_{filename}.{extension}"
    else:
        # Extension already in filename
        name, ext = os.path.splitext(filename)
        return f"{DATASET_NAME}_{model_name}_{name}{ext}"


def get_model_output_dir(model_name):
    """Get output directory for a specific model - creates if not exists"""
    model_dir = os.path.join(OUTPUT_DIR, model_name)
    os.makedirs(model_dir, exist_ok=True)
    return model_dir


# ============================================================================
# DATA LOADING
# ============================================================================
def detect_dataset_type(data_dir):
    """Detect dataset type from directory name"""
    dir_lower = data_dir.lower()
    if 'ciciot' in dir_lower:
        return 'ciciot2023'
    elif 'unsw' in dir_lower or 'nb15' in dir_lower:
        return 'unsw-nb15'
    elif 'ton' in dir_lower:
        return 'toniot'
    elif 'edge' in dir_lower or 'iiot' in dir_lower:
        return 'edge-iiot'
    return None


def load_dataset(data_dir):
    """Load dataset from directory with automatic header detection"""
    global DATASET_NAME

    print(f"\n{'='*70}")
    print(f" Loading Data from: {data_dir}")
    print("="*70)

    csv_files = list(Path(data_dir).glob("*.csv"))
    print(f"Found {len(csv_files)} CSV files")

    # Auto-detect dataset type first
    detected_type = detect_dataset_type(data_dir)
    if detected_type:
        DATASET_NAME = detected_type
        print(f"âœ“ Dataset Name: {DATASET_NAME}")

    # Get expected headers if defined
    expected_headers = None
    if detected_type and detected_type in TAXONOMY_MAPS:
        expected_headers = TAXONOMY_MAPS[detected_type].get('headers', None)

    dfs = []
    for f in tqdm(csv_files, desc="Loading"):
        try:
            # First, try loading with headers
            df = pd.read_csv(f, low_memory=False, nrows=5)
            first_col = str(df.columns[0])

            # Check if first column looks like data (IP address, number) instead of header
            has_header = True
            if '.' in first_col and any(c.isdigit() for c in first_col.split('.')[0]):
                # Looks like IP address - no header
                has_header = False
            elif first_col.replace('.', '').replace('-', '').isdigit():
                # Looks like a number - no header
                has_header = False

            # Reload with proper settings
            if has_header:
                df = pd.read_csv(f, low_memory=False)
            else:
                print(f"  âš  {f.name}: No header detected")
                df = pd.read_csv(f, header=None, low_memory=False)

                # Apply expected headers if available and column count matches
                if expected_headers and len(df.columns) == len(expected_headers):
                    df.columns = expected_headers
                    print(f"    âœ“ Applied standard headers ({len(expected_headers)} columns)")
                elif expected_headers and len(df.columns) == len(expected_headers) - 1:
                    df.columns = expected_headers[:-1]
                    print(f"    âœ“ Applied headers without last column ({len(df.columns)} columns)")
                else:
                    # Use generic column names
                    df.columns = [f'col_{i}' for i in range(len(df.columns))]
                    print(f"    âš  Using generic column names ({len(df.columns)} columns)")

            dfs.append(df)
        except Exception as e:
            print(f"  Error loading {f.name}: {e}")

    if not dfs:
        raise ValueError(f"No data loaded from {data_dir}")

    data = pd.concat(dfs, ignore_index=True)
    print(f"âœ“ Loaded {len(data):,} samples, {len(data.columns)} columns")
    print(f"  Columns: {list(data.columns[:10])}{'...' if len(data.columns) > 10 else ''}")

    return data


def preprocess_dataset(df, dataset_type=None):
    """Preprocess dataset based on its type"""
    global DATASET_NAME

    if dataset_type is None:
        dataset_type = DATASET_NAME

    print(f"âœ“ Detected dataset: {dataset_type}")

    if dataset_type not in TAXONOMY_MAPS:
        raise ValueError(f"Unknown dataset type: {dataset_type}")

    config = TAXONOMY_MAPS[dataset_type]
    label_col = config['label_col']

    # Auto-detect label column if not found
    if label_col not in df.columns:
        print(f"âš  Label column '{label_col}' not found. Searching alternatives...")
        # Use dataset-specific alternatives first, then generic
        alt_labels = config.get('alt_label_cols', [])
        possible_labels = alt_labels + ['attack_cat', 'Attack_cat', 'Label', 'label',
                          'class', 'Class', 'type', 'Type', 'Attack_type']
        found = False
        for col in possible_labels:
            if col in df.columns:
                label_col = col
                print(f"âœ“ Found alternative label column: '{label_col}'")
                found = True
                break

        if not found:
            # Show available columns
            print(f"Available columns: {list(df.columns)}")
            raise KeyError(f"Could not find label column. Tried: {possible_labels}")

    print(f"âœ“ Label column: {label_col}")

    # Clean label column - strip whitespace
    if df[label_col].dtype == 'object':
        df[label_col] = df[label_col].str.strip()

    # UNSW-NB15 special handling: NaN attack_cat = Normal traffic
    if dataset_type == 'unsw-nb15':
        # Fill NaN with 'Normal'
        df[label_col] = df[label_col].fillna('Normal')
        # Also check if 'Label' column exists (binary: 0=Normal, 1=Attack)
        if 'Label' in df.columns:
            # Where Label=0 and attack_cat is still weird, set to Normal
            df.loc[(df['Label'] == 0) & (~df[label_col].isin(config['taxonomy'].keys())), label_col] = 'Normal'
        print(f"âœ“ UNSW-NB15: Filled NaN attack_cat with 'Normal'")

    # Map labels to taxonomy
    df['Label_mapped'] = df[label_col].map(config['taxonomy'])

    # Handle unmapped labels
    unmapped = df[df['Label_mapped'].isna()][label_col].unique()
    unmapped_count = df['Label_mapped'].isna().sum()
    total_count = len(df)

    if unmapped_count > 0 and unmapped_count <= total_count * 0.1:
        # Less than 10% unmapped - drop them
        print(f"âš  Unmapped labels ({len(unmapped)}): {list(unmapped)[:10]}...")
        df = df.dropna(subset=['Label_mapped'])
        print(f"  Dropped {unmapped_count} rows with unmapped labels")
    elif unmapped_count > total_count * 0.1:
        # More than 10% unmapped - use original labels with cleaning
        print(f"âš  {unmapped_count}/{total_count} samples unmapped. Using cleaned original labels.")
        df['Label_mapped'] = df[label_col].astype(str).str.strip()
        # Remove 'nan' string
        df = df[df['Label_mapped'] != 'nan']
        df = df[df['Label_mapped'] != '']

    # Balance classes
    class_counts = df['Label_mapped'].value_counts()
    print(f"\nClass Distribution:")
    print(class_counts)

    # Limit samples per class
    max_samples = 50000
    min_samples = 1000

    balanced_dfs = []
    for label in class_counts.index:
        class_df = df[df['Label_mapped'] == label]
        n = len(class_df)
        if n > max_samples:
            class_df = class_df.sample(max_samples, random_state=CONFIG['random_state'])
        balanced_dfs.append(class_df)

    df = pd.concat(balanced_dfs, ignore_index=True)

    # Get features - filter drop_cols to only existing columns
    drop_cols = [c for c in config['drop_cols'] if c in df.columns] + ['Label_mapped']
    feature_cols = [c for c in df.columns if c not in drop_cols and c != label_col]

    # Select numeric columns only
    numeric_cols = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()

    X = df[numeric_cols].values
    y = df['Label_mapped'].values

    print(f"âœ“ Features: {len(numeric_cols)}")

    return X, y, numeric_cols


def prepare_data(X, y):
    """Prepare data with proper train/val/test split to avoid data leakage"""
    print(f"\n{'â”€'*50}")
    print(" Preprocessing (NO Data Leakage)")
    print("â”€"*50)

    # First split: train+val vs test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=CONFIG['test_size'],
        stratify=y, random_state=CONFIG['random_state']
    )

    # Second split: train vs val
    val_ratio = CONFIG['val_size'] / (1 - CONFIG['test_size'])
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_ratio,
        stratify=y_temp, random_state=CONFIG['random_state']
    )

    print(f"  Split: Train={len(X_train):,}, Val={len(X_val):,}, Test={len(X_test):,}")

    # Encode labels
    le = LabelEncoder()
    y_train = le.fit_transform(y_train)
    y_val = le.transform(y_val)
    y_test = le.transform(y_test)

    n_classes = len(le.classes_)
    print(f"  Classes ({n_classes}): {list(le.classes_)}")

    # Replace inf values with NaN (will be imputed)
    X_train = np.where(np.isinf(X_train), np.nan, X_train)
    X_val = np.where(np.isinf(X_val), np.nan, X_val)
    X_test = np.where(np.isinf(X_test), np.nan, X_test)

    # Also clip very large values
    max_val = np.finfo(np.float64).max / 10
    X_train = np.clip(X_train, -max_val, max_val)
    X_val = np.clip(X_val, -max_val, max_val)
    X_test = np.clip(X_test, -max_val, max_val)

    # Impute missing values - fit on train only
    imputer = SimpleImputer(strategy='median')
    X_train = imputer.fit_transform(X_train)
    X_val = imputer.transform(X_val)
    X_test = imputer.transform(X_test)

    # Scale - fit on train only
    scaler = RobustScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    # Feature selection - fit on train only
    print(f"\n  Feature Selection (Top {CONFIG['n_features']})...")
    from sklearn.feature_selection import mutual_info_classif
    mi_scores = mutual_info_classif(X_train, y_train, random_state=CONFIG['random_state'])
    top_indices = np.argsort(mi_scores)[-CONFIG['n_features']:]

    X_train = X_train[:, top_indices]
    X_val = X_val[:, top_indices]
    X_test = X_test[:, top_indices]

    # Get feature names (we'll use generic names if not available)
    feature_names = [f"feature_{i}" for i in range(X_train.shape[1])]
    print(f"  âœ“ Selected {X_train.shape[1]} features")

    # Apply SMOTE on training data only
    print(f"\n  Applying SMOTE...")
    smote = SMOTE(random_state=CONFIG['random_state'])
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    print(f"  âœ“ SMOTE: {len(X_train):,} â†’ {len(X_train_resampled):,}")

    return {
        'X_train': X_train_resampled,
        'y_train': y_train_resampled,
        'X_val': X_val,
        'y_val': y_val,
        'X_test': X_test,
        'y_test': y_test,
        'le': le,
        'n_classes': n_classes,
        'n_features': X_train.shape[1],
        'feature_names': feature_names,
    }


def load_and_prepare_data():
    """Wrapper function to load dataset and prepare data"""
    df = load_dataset(DATA_DIR)
    X, y, feature_cols = preprocess_dataset(df)
    data = prepare_data(X, y)
    data['feature_names'] = feature_cols[:data['n_features']]  # Use actual feature names
    return data


# ============================================================================
# EVALUATION
# ============================================================================
def evaluate_model(y_true, y_pred, le):
    """Evaluate model and return metrics dict"""
    results = {
        'accuracy': accuracy_score(y_true, y_pred),
        'f1_macro': f1_score(y_true, y_pred, average='macro'),
        'f1_weighted': f1_score(y_true, y_pred, average='weighted'),
        'precision_macro': precision_score(y_true, y_pred, average='macro', zero_division=0),
        'recall_macro': recall_score(y_true, y_pred, average='macro', zero_division=0),
        'mcc': matthews_corrcoef(y_true, y_pred),
        'balanced_accuracy': balanced_accuracy_score(y_true, y_pred),
    }

    # Per-class metrics
    report = classification_report(y_true, y_pred, target_names=le.classes_, output_dict=True)
    results['per_class'] = {
        cls: {
            'precision': report[cls]['precision'],
            'recall': report[cls]['recall'],
            'f1': report[cls]['f1-score'],
            'support': report[cls]['support']
        }
        for cls in le.classes_
    }

    results['confusion_matrix'] = confusion_matrix(y_true, y_pred).tolist()
    results['classification_report'] = classification_report(y_true, y_pred, target_names=le.classes_)

    return results


# ============================================================================
# DEEP LEARNING MODELS - MODE COLLAPSE FIX
# ============================================================================
class FocalLoss(nn.Module):
    """Focal Loss - mode collapse'a karÅŸÄ± etkili"""
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha  # class weights
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(inputs, targets, weight=self.alpha, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_state = None

    def __call__(self, score, model):
        if self.best_score is None:
            self.best_score = score
            self.best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            self.counter = 0


class ImprovedMLP(nn.Module):
    """MLP with proper initialization for avoiding mode collapse"""
    def __init__(self, input_dim, num_classes, hidden_dims=[256, 128], dropout=0.3):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(),  # ReLU yerine GELU kullanÄ±labilir
                nn.Dropout(dropout)
            ])
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, num_classes))
        self.model = nn.Sequential(*layers)

        # Proper weight initialization
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.model(x)


class Improved1DCNN(nn.Module):
    """1D CNN with proper initialization"""
    def __init__(self, input_dim, num_classes, num_filters=64, dropout=0.3):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv1d(1, num_filters, kernel_size=3, padding=1),
            nn.BatchNorm1d(num_filters),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(num_filters, num_filters * 2, kernel_size=3, padding=1),
            nn.BatchNorm1d(num_filters * 2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        self.fc = nn.Sequential(
            nn.Linear(num_filters * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        x = self.conv_layers(x)
        x = x.squeeze(-1)
        return self.fc(x)


class ImprovedLSTM(nn.Module):
    """LSTM with proper initialization for tabular data"""
    def __init__(self, input_dim, num_classes, hidden_size=128, num_layers=2, dropout=0.3):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_classes)
        )
        self._init_weights()

    def _init_weights(self):
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.kaiming_normal_(param, mode='fan_out', nonlinearity='relu')
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)
        for m in self.fc.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # x: (batch, features) -> (batch, seq_len=1, features)
        if x.dim() == 2:
            x = x.unsqueeze(1)
        lstm_out, _ = self.lstm(x)
        # Take last time step
        out = lstm_out[:, -1, :]
        return self.fc(out)


class ImprovedGRU(nn.Module):
    """GRU with proper initialization for tabular data"""
    def __init__(self, input_dim, num_classes, hidden_size=128, num_layers=2, dropout=0.3):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_classes)
        )
        self._init_weights()

    def _init_weights(self):
        for name, param in self.gru.named_parameters():
            if 'weight_ih' in name:
                nn.init.kaiming_normal_(param, mode='fan_out', nonlinearity='relu')
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)
        for m in self.fc.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        gru_out, _ = self.gru(x)
        out = gru_out[:, -1, :]
        return self.fc(out)


class ImprovedCNN_LSTM(nn.Module):
    """CNN + LSTM hybrid with proper initialization"""
    def __init__(self, input_dim, num_classes, num_filters=64, hidden_size=128, dropout=0.3):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv1d(1, num_filters, kernel_size=3, padding=1),
            nn.BatchNorm1d(num_filters),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.lstm = nn.LSTM(
            input_size=num_filters,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=False
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_classes)
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.cnn.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.kaiming_normal_(param, mode='fan_out', nonlinearity='relu')
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)
        for m in self.fc.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # x: (batch, features)
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (batch, 1, features)

        # CNN expects (batch, channels, length)
        x = self.cnn(x)  # (batch, num_filters, features)

        # LSTM expects (batch, seq_len, features)
        x = x.permute(0, 2, 1)  # (batch, features, num_filters)

        lstm_out, _ = self.lstm(x)
        out = lstm_out[:, -1, :]
        return self.fc(out)


class ImprovedTransformer(nn.Module):
    """Transformer encoder for tabular data with proper initialization"""
    def __init__(self, input_dim, num_classes, d_model=64, nhead=4, num_layers=2, dropout=0.3):
        super().__init__()

        # Ensure d_model is divisible by nhead
        self.d_model = d_model

        # Input projection
        self.input_proj = nn.Linear(input_dim, d_model)

        # Positional encoding (learnable)
        self.pos_embedding = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation='relu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output layers
        self.fc = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes)
        )
        self._init_weights()

    def _init_weights(self):
        # Initialize input projection
        nn.init.kaiming_normal_(self.input_proj.weight, mode='fan_out', nonlinearity='relu')
        nn.init.constant_(self.input_proj.bias, 0)

        # Initialize transformer layers
        for name, param in self.transformer.named_parameters():
            if 'weight' in name and param.dim() >= 2:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)

        # Initialize FC layers
        for m in self.fc.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # x: (batch, features)
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (batch, 1, features)

        # Project to d_model dimensions
        x = self.input_proj(x)  # (batch, 1, d_model)

        # Add positional embedding
        x = x + self.pos_embedding

        # Transformer encoding
        x = self.transformer(x)  # (batch, 1, d_model)

        # Take the output (only one position)
        x = x[:, 0, :]  # (batch, d_model)

        return self.fc(x)


def check_mode_collapse(y_pred, n_classes, threshold=0.90):
    """Check if model has mode collapse"""
    unique, counts = np.unique(y_pred, return_counts=True)
    total = len(y_pred)
    max_ratio = counts.max() / total
    dominant_class = unique[counts.argmax()]

    is_collapsed = max_ratio > threshold or len(unique) < n_classes * 0.5

    collapse_info = {
        'is_collapsed': is_collapsed,
        'dominant_class': int(dominant_class),
        'dominant_ratio': float(max_ratio),
        'num_predicted_classes': len(unique),
        'expected_classes': n_classes
    }

    return is_collapsed, collapse_info


def train_dl_model(model, data, epochs=100, lr=0.001, batch_size=512, patience=15, weight_decay=0.01, use_focal_loss=True):
    """Train DL model with mode collapse prevention - V2 with warmup"""
    model = model.to(DEVICE)

    train_dataset = TensorDataset(
        torch.FloatTensor(data['X_train']),
        torch.LongTensor(data['y_train'])
    )
    val_dataset = TensorDataset(
        torch.FloatTensor(data['X_val']),
        torch.LongTensor(data['y_val'])
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # Class weights from TRAINING set (SMOTE-balanced, so should be equal)
    train_class_counts = np.bincount(data['y_train'], minlength=data['n_classes'])
    train_class_counts = np.maximum(train_class_counts, 1)

    # More aggressive class weights
    total_samples = len(data['y_train'])
    class_weights = torch.FloatTensor(
        total_samples / (data['n_classes'] * train_class_counts)
    ).to(DEVICE)

    # Normalize weights to prevent exploding gradients
    class_weights = class_weights / class_weights.sum() * data['n_classes']

    # Loss function
    if use_focal_loss:
        criterion = FocalLoss(alpha=class_weights, gamma=2.0)
    else:
        criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Warmup + ReduceLROnPlateau
    warmup_epochs = 5
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, min_lr=1e-6)
    early_stopping = EarlyStopping(patience=patience)

    best_f1 = 0
    best_model_state = None

    for epoch in range(epochs):
        # Warmup: linearly increase LR for first few epochs
        if epoch < warmup_epochs:
            warmup_lr = lr * (epoch + 1) / warmup_epochs
            for param_group in optimizer.param_groups:
                param_group['lr'] = warmup_lr

        # Train
        model.train()
        epoch_loss = 0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(DEVICE), batch_y.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()

        # Validate
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X = batch_X.to(DEVICE)
                outputs = model(batch_X)
                preds = outputs.argmax(dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(batch_y.numpy())

        val_f1 = f1_score(all_labels, all_preds, average='macro')

        # Only use scheduler after warmup
        if epoch >= warmup_epochs:
            scheduler.step(val_f1)

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_model_state = model.state_dict().copy()

        early_stopping(val_f1, model)
        if early_stopping.early_stop:
            break

    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    elif early_stopping.best_state:
        model.load_state_dict(early_stopping.best_state)

    return model, best_f1


def optimize_dl_model(model_class, model_name, data, n_trials=30):
    """Optimize DL model with mode collapse prevention - V2"""
    print(f"  Optimizing {model_name} with Optuna ({n_trials} trials)...")

    def objective(trial):
        # Hyperparameters based on successful LSTM config
        lr = trial.suggest_float('lr', 1e-4, 2e-3, log=True)  # Lower range
        dropout = trial.suggest_float('dropout', 0.1, 0.3)  # Lower dropout (LSTM used 0.13)
        batch_size = trial.suggest_categorical('batch_size', [512, 1024, 2048])  # Larger batches
        weight_decay = trial.suggest_float('weight_decay', 1e-5, 5e-4, log=True)

        if model_name == 'MLP':
            hidden_dim = trial.suggest_categorical('hidden_dim', [128, 256, 512])
            num_layers = trial.suggest_int('num_layers', 2, 3)
            hidden_dims = [hidden_dim // (2**i) for i in range(num_layers)]
            hidden_dims = [max(32, h) for h in hidden_dims]
            model = model_class(
                input_dim=data['n_features'],
                num_classes=data['n_classes'],
                hidden_dims=hidden_dims,
                dropout=dropout
            )

        elif model_name == 'CNN_1D':
            num_filters = trial.suggest_categorical('num_filters', [32, 64, 128])
            model = model_class(
                input_dim=data['n_features'],
                num_classes=data['n_classes'],
                num_filters=num_filters,
                dropout=dropout
            )

        elif model_name == 'LSTM':
            hidden_size = trial.suggest_categorical('hidden_size', [128, 256, 512])
            num_layers = trial.suggest_int('num_layers', 1, 2)
            model = model_class(
                input_dim=data['n_features'],
                num_classes=data['n_classes'],
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout
            )

        elif model_name == 'GRU':
            hidden_size = trial.suggest_categorical('hidden_size', [128, 256, 512])
            num_layers = trial.suggest_int('num_layers', 1, 2)
            model = model_class(
                input_dim=data['n_features'],
                num_classes=data['n_classes'],
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout
            )

        elif model_name == 'CNN_LSTM':
            num_filters = trial.suggest_categorical('num_filters', [32, 64, 128])
            hidden_size = trial.suggest_categorical('hidden_size', [64, 128])
            model = model_class(
                input_dim=data['n_features'],
                num_classes=data['n_classes'],
                num_filters=num_filters,
                hidden_size=hidden_size,
                dropout=dropout
            )

        elif model_name == 'Transformer':
            d_model = trial.suggest_categorical('d_model', [64, 128])
            nhead = trial.suggest_categorical('nhead', [4, 8])
            num_layers = trial.suggest_int('num_layers', 1, 2)
            # Ensure d_model is divisible by nhead
            if d_model % nhead != 0:
                d_model = (d_model // nhead) * nhead
            model = model_class(
                input_dim=data['n_features'],
                num_classes=data['n_classes'],
                d_model=d_model,
                nhead=nhead,
                num_layers=num_layers,
                dropout=dropout
            )

        else:
            raise ValueError(f"Unknown model: {model_name}")

        try:
            model, val_f1 = train_dl_model(
                model, data,
                epochs=50,
                lr=lr,
                batch_size=batch_size,
                patience=10,
                weight_decay=weight_decay,
                use_focal_loss=True
            )

            # Check for mode collapse
            model.eval()
            with torch.no_grad():
                X_val_tensor = torch.FloatTensor(data['X_val']).to(DEVICE)
                outputs = model(X_val_tensor)
                y_pred = outputs.argmax(dim=1).cpu().numpy()

            is_collapsed, _ = check_mode_collapse(y_pred, data['n_classes'])
            if is_collapsed:
                return 0.0

            return val_f1

        except Exception as e:
            print(f"    Trial failed: {e}")
            return 0.0

    study = optuna.create_study(
        direction='maximize',
        sampler=TPESampler(seed=CONFIG['random_state'])
    )

    # Early stopping callback - eÄŸer Ã¶ÄŸrenme olmuyorsa durdur
    def early_stopping_callback(study, trial):
        """Stop optimization if no learning is happening"""
        n_trials_check = 10  # Ä°lk 10 trial'dan sonra kontrol et
        min_acceptable_f1 = 0.10  # En az %10 F1 olmalÄ±

        if trial.number >= n_trials_check:
            # Son n_trials_check trial'Ä±n best value'su
            if study.best_value < min_acceptable_f1:
                print(f"\n  âš  Early stopping: No learning after {trial.number + 1} trials (best F1: {study.best_value:.4f})")
                print(f"    Model appears unsuitable for this dataset. Skipping...")
                study.stop()

    study.optimize(objective, n_trials=n_trials, show_progress_bar=True, callbacks=[early_stopping_callback])

    best_params = dict(study.best_trial.params)
    best_params['lr'] = best_params.get('lr', 1e-3)
    best_params['dropout'] = best_params.get('dropout', 0.3)
    best_params['batch_size'] = best_params.get('batch_size', 512)
    best_params['weight_decay'] = best_params.get('weight_decay', 1e-4)

    return best_params, study.best_value


# ============================================================================
# ML MODEL OPTIMIZERS
# ============================================================================
def optimize_xgboost(data, n_trials=50):
    """Optimize XGBoost with Optuna"""

    # Check GPU availability with simple test data
    use_gpu = False
    if torch.cuda.is_available():
        try:
            import numpy as np
            X_test_gpu = np.random.rand(100, 10).astype(np.float32)
            y_test_gpu = np.random.randint(0, 10, 100)
            test_model = XGBClassifier(tree_method='hist', device='cuda', n_estimators=5, verbosity=0)
            test_model.fit(X_test_gpu, y_test_gpu)
            use_gpu = True
            print("  âœ“ XGBoost GPU (CUDA) enabled")
        except Exception as e:
            print(f"  âš  XGBoost GPU not available: {str(e)[:50]}")
            use_gpu = False
    else:
        print("  â„¹ CUDA not available, using CPU")

    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'max_depth': trial.suggest_int('max_depth', 4, 12),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
            'random_state': CONFIG['random_state'],
            'verbosity': 0,
        }
        if use_gpu:
            params['tree_method'] = 'hist'
            params['device'] = 'cuda'

        try:
            model = XGBClassifier(**params)
            model.fit(data['X_train'], data['y_train'], eval_set=[(data['X_val'], data['y_val'])], verbose=False)
            y_pred = model.predict(data['X_val'])
            return f1_score(data['y_val'], y_pred, average='macro')
        except Exception as e:
            print(f"    Trial failed: {str(e)[:80]}")
            return 0.0

    study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=CONFIG['random_state']))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True, catch=(Exception,))

    best_params = dict(study.best_params)
    best_params['_use_gpu'] = use_gpu
    return best_params, study.best_value


def optimize_lightgbm(data, n_trials=50):
    """Optimize LightGBM with Optuna"""
    from lightgbm import LGBMClassifier

    # Check if GPU version is available (works on Ubuntu/WSL)
    use_gpu = False
    if torch.cuda.is_available():
        try:
            import numpy as np
            X_test_gpu = np.random.rand(100, 10).astype(np.float32)
            y_test_gpu = np.random.randint(0, 10, 100)
            test_model = LGBMClassifier(device='gpu', n_estimators=5, verbose=-1)
            test_model.fit(X_test_gpu, y_test_gpu)
            use_gpu = True
            print("  âœ“ LightGBM GPU enabled")
        except Exception as e:
            print(f"  â„¹ LightGBM using CPU: {str(e)[:50]}")
            use_gpu = False
    else:
        print("  â„¹ LightGBM using CPU")

    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'max_depth': trial.suggest_int('max_depth', 4, 12),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 20, 150),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
            'verbose': -1,
            'random_state': CONFIG['random_state'],
            'n_jobs': -1,
        }
        if use_gpu:
            params['device'] = 'gpu'

        try:
            model = LGBMClassifier(**params)
            model.fit(data['X_train'], data['y_train'], eval_set=[(data['X_val'], data['y_val'])])
            y_pred = model.predict(data['X_val'])
            return f1_score(data['y_val'], y_pred, average='macro')
        except Exception as e:
            print(f"    Trial failed: {str(e)[:50]}...")
            return 0.0

    study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=CONFIG['random_state']))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True, catch=(Exception,))

    best_params = dict(study.best_params)
    best_params['_use_gpu'] = use_gpu
    return best_params, study.best_value


def optimize_random_forest(data, n_trials=50):
    """Optimize Random Forest with Optuna"""
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'max_depth': trial.suggest_int('max_depth', 10, 30),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
            'random_state': CONFIG['random_state'],
            'n_jobs': -1,
        }
        try:
            model = RandomForestClassifier(**params)
            model.fit(data['X_train'], data['y_train'])
            y_pred = model.predict(data['X_val'])
            return f1_score(data['y_val'], y_pred, average='macro')
        except Exception as e:
            print(f"    Trial failed: {e}")
            return 0.0

    study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=CONFIG['random_state']))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True, catch=(Exception,))
    return study.best_params, study.best_value


def optimize_svm(data, n_trials=50):
    """Optimize SVM with Optuna - using subset"""
    n_subset = min(50000, len(data['X_train']))
    subset_idx = np.random.choice(len(data['X_train']), n_subset, replace=False)
    X_train_subset = data['X_train'][subset_idx]
    y_train_subset = data['y_train'][subset_idx]
    print(f"  âš  SVM training on subset: {n_subset:,} samples")

    def objective(trial):
        C = trial.suggest_float('C', 0.1, 100.0, log=True)
        gamma = trial.suggest_float('gamma', 1e-4, 1.0, log=True)
        try:
            model = SVC(C=C, gamma=gamma, kernel='rbf', random_state=CONFIG['random_state'],
                        probability=True, class_weight='balanced')
            model.fit(X_train_subset, y_train_subset)
            y_pred = model.predict(data['X_val'])
            return f1_score(data['y_val'], y_pred, average='macro')
        except Exception as e:
            print(f"    Trial failed: {e}")
            return 0.0

    study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=CONFIG['random_state']))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True, catch=(Exception,))

    best_params = {
        'C': study.best_params['C'],
        'gamma': study.best_params['gamma'],
        'kernel': 'rbf',
        'random_state': CONFIG['random_state'],
        'probability': True,
        'class_weight': 'balanced',
    }
    return best_params, study.best_value


def optimize_catboost(data, n_trials=50):
    """Optimize CatBoost with Optuna"""
    from catboost import CatBoostClassifier

    # Check GPU availability with simple test data
    use_gpu = False
    if torch.cuda.is_available():
        try:
            import numpy as np
            X_test_gpu = np.random.rand(100, 10).astype(np.float32)
            y_test_gpu = np.random.randint(0, 10, 100)
            test_model = CatBoostClassifier(task_type='GPU', iterations=5, verbose=False)
            test_model.fit(X_test_gpu, y_test_gpu, verbose=False)
            use_gpu = True
            print("  âœ“ CatBoost GPU enabled")
        except Exception as e:
            print(f"  âš  CatBoost GPU not available: {str(e)[:50]}")
            use_gpu = False
    else:
        print("  â„¹ CatBoost using CPU")

    def objective(trial):
        params = {
            'iterations': trial.suggest_int('iterations', 100, 500),
            'depth': trial.suggest_int('depth', 4, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-3, 10.0, log=True),
            'border_count': trial.suggest_int('border_count', 32, 255),
            'random_seed': CONFIG['random_state'],
            'verbose': False,
            'task_type': 'GPU' if use_gpu else 'CPU',
            'auto_class_weights': 'Balanced',
        }

        try:
            model = CatBoostClassifier(**params)
            model.fit(data['X_train'], data['y_train'], eval_set=(data['X_val'], data['y_val']), verbose=False)
            y_pred = model.predict(data['X_val'])
            return f1_score(data['y_val'], y_pred, average='macro')
        except Exception as e:
            print(f"    Trial failed: {str(e)[:80]}")
            return 0.0

    study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=CONFIG['random_state']))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True, catch=(Exception,))

    best_params = dict(study.best_params)
    best_params['_use_gpu'] = use_gpu
    return best_params, study.best_value


# ============================================================================
# SHAP & LIME EXPLANATIONS
# ============================================================================
def compute_shap_explanations(model, X_train, X_test, feature_names, model_name, output_dir):
    """Compute SHAP explanations"""
    if not SHAP_AVAILABLE:
        return None, "SHAP not available"

    # CatBoost SHAP crashes on Windows only - works on Linux/Ubuntu
    import platform
    if 'CatBoost' in model_name and platform.system() == 'Windows':
        print(f"  âš  Skipping SHAP for {model_name} (Windows compatibility)")
        return None, "CatBoost SHAP skipped on Windows"

    print(f"  Computing SHAP for {model_name}...")
    try:
        n_samples = min(CONFIG['shap_samples'], len(X_test))
        idx = np.random.choice(len(X_test), n_samples, replace=False)
        X_sample = np.asarray(X_test[idx], dtype=np.float32)
        X_background = np.asarray(X_train[:min(100, len(X_train))], dtype=np.float32)

        tree_models = ['XGB', 'LightGBM', 'Random_Forest', 'Decision', 'CatBoost']
        is_tree_model = any(tm in model_name for tm in tree_models)

        if is_tree_model:
            if 'XGB' in model_name:
                # XGBoost + SHAP has known compatibility issues with multi-class base_score
                # Use function-based explainer which is more robust
                print("    Using function-based SHAP for XGBoost (avoids base_score bug)...")

                # Create a wrapper function for predict_proba
                def xgb_predict_proba(X):
                    return model.predict_proba(X)

                # Use Explainer with the prediction function - works reliably
                explainer = shap.Explainer(xgb_predict_proba, X_background, algorithm='auto')
                shap_explanation = explainer(X_sample)

                # Extract SHAP values from Explanation object
                shap_values = shap_explanation.values
                print(f"    âœ“ Function-based SHAP computed")
            elif 'LightGBM' in model_name or 'CatBoost' in model_name:
                # LightGBM and CatBoost work better with TreeExplainer
                try:
                    explainer = shap.TreeExplainer(model)
                    shap_values = explainer.shap_values(X_sample)
                except Exception as e:
                    print(f"    TreeExplainer failed, using function-based: {str(e)[:50]}")
                    def lgb_predict_proba(X):
                        return model.predict_proba(X)
                    explainer = shap.Explainer(lgb_predict_proba, X_background)
                    shap_explanation = explainer(X_sample)
                    shap_values = shap_explanation.values
            else:
                # Random Forest, Decision Tree - TreeExplainer works fine
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X_sample)
        else:
            explainer = shap.KernelExplainer(model.predict_proba, X_background)
            shap_values = explainer.shap_values(X_sample, nsamples=100)

        # Handle multi-class SHAP values - more robust handling
        shap_values_arr = np.array(shap_values)
        print(f"    SHAP values shape: {shap_values_arr.shape}")

        if isinstance(shap_values, list):
            # Multi-class: list of (n_samples, n_features) arrays
            # Stack and take mean absolute value across classes and samples
            stacked = np.stack(shap_values, axis=0)  # (n_classes, n_samples, n_features)
            shap_values_mean = np.abs(stacked).mean(axis=(0, 1))  # (n_features,)
        elif shap_values_arr.ndim == 3:
            # Shape: (n_samples, n_features, n_classes) or (n_classes, n_samples, n_features)
            if shap_values_arr.shape[0] == n_samples:
                # (n_samples, n_features, n_classes)
                shap_values_mean = np.abs(shap_values_arr).mean(axis=(0, 2))
            else:
                # (n_classes, n_samples, n_features)
                shap_values_mean = np.abs(shap_values_arr).mean(axis=(0, 1))
        elif shap_values_arr.ndim == 2:
            # Binary or single output: (n_samples, n_features)
            shap_values_mean = np.abs(shap_values_arr).mean(axis=0)
        else:
            raise ValueError(f"Unexpected SHAP values shape: {shap_values_arr.shape}")

        # Ensure correct length
        if len(shap_values_mean) != len(feature_names):
            raise ValueError(f"SHAP values length ({len(shap_values_mean)}) != features ({len(feature_names)})")

        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': shap_values_mean
        }).sort_values('importance', ascending=False)

        importance_df.to_csv(f"{output_dir}/{DATASET_NAME}_{model_name}_shap_importance.csv", index=False)
        print(f"    âœ“ SHAP importance saved")

        # Plot importance
        plt.figure(figsize=(10, 8))
        sns.barplot(data=importance_df.head(15), y='feature', x='importance', palette='viridis')
        plt.title(f'SHAP Feature Importance - {model_name}')
        plt.xlabel('Mean |SHAP Value|')
        plt.tight_layout()
        plt.savefig(f"{output_dir}/{DATASET_NAME}_{model_name}_shap_importance.png", dpi=150)
        plt.close()

        # Summary plot - use first class for multi-class
        plt.figure(figsize=(10, 8))
        if isinstance(shap_values, list):
            shap.summary_plot(shap_values[0], X_sample, feature_names=feature_names, show=False)
        elif shap_values_arr.ndim == 3:
            if shap_values_arr.shape[0] == n_samples:
                shap.summary_plot(shap_values_arr[:, :, 0], X_sample, feature_names=feature_names, show=False)
            else:
                shap.summary_plot(shap_values_arr[0], X_sample, feature_names=feature_names, show=False)
        else:
            shap.summary_plot(shap_values, X_sample, feature_names=feature_names, show=False)
        plt.title(f'SHAP Summary - {model_name}')
        plt.tight_layout()
        plt.savefig(f"{output_dir}/{DATASET_NAME}_{model_name}_shap_summary.png", dpi=150)
        plt.close()
        print(f"    âœ“ SHAP plots saved")

        return importance_df, None

    except Exception as e:
        error_msg = f"{type(e).__name__}: {str(e)}"
        print(f"    âœ— SHAP failed: {error_msg}")
        import traceback
        traceback.print_exc()
        return None, error_msg


def compute_lime_explanations(model, X_train, X_test, y_test, feature_names, le, model_name, output_dir, n_samples=5):
    """Compute LIME explanations"""
    if not LIME_AVAILABLE:
        return None, "LIME not available"

    print(f"  Computing LIME for {model_name}...")
    try:
        explainer = lime.lime_tabular.LimeTabularExplainer(
            X_train,
            feature_names=feature_names,
            class_names=le.classes_,
            mode='classification'
        )

        # Get diverse samples
        unique_classes = np.unique(y_test)
        sample_indices = []
        for cls in unique_classes[:n_samples]:
            cls_idx = np.where(y_test == cls)[0]
            if len(cls_idx) > 0:
                sample_indices.append(cls_idx[0])

        explanations = []
        fig, axes = plt.subplots(len(sample_indices), 1, figsize=(12, 4*len(sample_indices)))
        if len(sample_indices) == 1:
            axes = [axes]

        for i, idx in enumerate(sample_indices):
            if hasattr(model, 'predict_proba'):
                exp = explainer.explain_instance(X_test[idx], model.predict_proba, num_features=10)
            else:
                continue

            pred_class = model.predict([X_test[idx]])[0]
            true_class = y_test[idx]

            explanation_data = {
                'sample_idx': int(idx),
                'true_class': le.classes_[true_class],
                'predicted_class': le.classes_[pred_class],
                'explanation': exp.as_list()
            }
            explanations.append(explanation_data)

            # Plot
            exp_list = exp.as_list()
            features = [x[0] for x in exp_list]
            weights = [x[1] for x in exp_list]
            colors = ['green' if w > 0 else 'red' for w in weights]

            axes[i].barh(features, weights, color=colors)
            axes[i].set_xlabel('Contribution')
            axes[i].set_title(f'Sample {idx}: True={le.classes_[true_class]}, Pred={le.classes_[pred_class]}')

        plt.tight_layout()
        plt.savefig(f"{output_dir}/{DATASET_NAME}_{model_name}_lime_explanations.png", dpi=150)
        plt.close()

        with open(f"{output_dir}/{DATASET_NAME}_{model_name}_lime_results.json", 'w', encoding='utf-8') as f:
            json.dump(explanations, f, indent=2)

        print(f"    âœ“ LIME explanations saved")
        return explanations, None

    except Exception as e:
        error_msg = f"{type(e).__name__}: {str(e)}"
        print(f"    âœ— LIME failed: {error_msg}")
        return None, error_msg


# ============================================================================
# PLOTTING FUNCTIONS
# ============================================================================
def plot_confusion_matrix(cm, classes, model_name, output_dir):
    """Plot normalized confusion matrix"""
    plt.figure(figsize=(12, 10))
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_normalized = np.nan_to_num(cm_normalized)

    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.title(f'Confusion Matrix (Normalized) - {model_name}')
    plt.ylabel('True')
    plt.xlabel('Predicted')
    plt.tight_layout()
    model_dir = get_model_output_dir(model_name)
    plt.savefig(f"{model_dir}/{DATASET_NAME}_{model_name}_confusion_matrix.png", dpi=150)
    plt.close()


def generate_summary_report(model_name, results, best_params, training_time, output_dir,
                           data, shap_importance=None, lime_results=None, collapse_info=None,
                           shap_error=None, lime_error=None):
    """Generate comprehensive summary report"""
    report = []
    report.append("=" * 80)
    report.append(f"EXPERIMENT SUMMARY - {model_name}")
    report.append("=" * 80)
    report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"Dataset: {DATASET_NAME}")
    report.append("")

    # Model Configuration
    report.append("-" * 80)
    report.append("1. MODEL CONFIGURATION")
    report.append("-" * 80)
    report.append(f"Model: {model_name}")
    report.append(f"Parameters:\n{json.dumps(best_params, indent=2)}")
    report.append("")

    # Dataset Info
    report.append("-" * 80)
    report.append("2. DATASET INFORMATION")
    report.append("-" * 80)
    report.append(f"Training samples: {len(data['X_train']):,}")
    report.append(f"Validation samples: {len(data['X_val']):,}")
    report.append(f"Test samples: {len(data['X_test']):,}")
    report.append(f"Number of features: {data['n_features']}")
    report.append(f"Number of classes: {data['n_classes']}")
    report.append(f"Classes: {list(data['le'].classes_)}")
    report.append("")

    # Performance Metrics
    report.append("-" * 80)
    report.append("3. PERFORMANCE METRICS")
    report.append("-" * 80)
    report.append(f"Accuracy:           {results['accuracy']:.4f}")
    report.append(f"F1-Score (Macro):   {results['f1_macro']:.4f}")
    report.append(f"F1-Score (Weighted):{results['f1_weighted']:.4f}")
    report.append(f"Precision (Macro):  {results['precision_macro']:.4f}")
    report.append(f"Recall (Macro):     {results['recall_macro']:.4f}")
    report.append(f"MCC:                {results['mcc']:.4f}")
    report.append(f"Balanced Accuracy:  {results['balanced_accuracy']:.4f}")
    report.append(f"Training Time:      {training_time:.2f} seconds")
    report.append("")

    # Mode Collapse Warning
    if collapse_info and collapse_info.get('is_collapsed', False):
        report.append("-" * 80)
        report.append("âš ï¸ MODE COLLAPSE DETECTED âš ï¸")
        report.append("-" * 80)
        report.append(f"Dominant Class: {collapse_info['dominant_class']}")
        report.append(f"Dominant Ratio: {collapse_info['dominant_ratio']:.2%}")
        report.append(f"Predicted Classes: {collapse_info['num_predicted_classes']}/{collapse_info['expected_classes']}")
        report.append("")

    # Per-class Performance
    report.append("-" * 80)
    report.append("4. PER-CLASS PERFORMANCE")
    report.append("-" * 80)
    report.append(f"{'Class':<20} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'Support':>10}")
    report.append("-" * 60)
    for cls, metrics in results['per_class'].items():
        report.append(f"{cls:<20} {metrics['precision']:>10.4f} {metrics['recall']:>10.4f} {metrics['f1']:>10.4f} {metrics['support']:>10}")
    report.append("")

    # Classification Report
    report.append("-" * 80)
    report.append("5. CLASSIFICATION REPORT")
    report.append("-" * 80)
    report.append(results['classification_report'])
    report.append("")

    # SHAP Importance
    report.append("-" * 80)
    report.append("6. FEATURE IMPORTANCE (SHAP)")
    report.append("-" * 80)
    if shap_importance is not None:
        for i, row in shap_importance.head(15).iterrows():
            report.append(f"  {i+1:>2}. {row['feature']:<25} {row['importance']:.6f}")
    elif shap_error:
        report.append(f"  âš  SHAP not available: {shap_error}")
    else:
        report.append("  âš  SHAP not computed")
    report.append("")

    # LIME Results
    report.append("-" * 80)
    report.append("7. LIME EXPLANATIONS")
    report.append("-" * 80)
    if lime_results:
        report.append(f"LIME explanations generated for {len(lime_results)} samples")
        for exp in lime_results:
            report.append(f"  Sample {exp['sample_idx']}: True={exp['true_class']}, Pred={exp['predicted_class']}")
    elif lime_error:
        report.append(f"  âš  LIME not available: {lime_error}")
    else:
        report.append("  âš  LIME not computed")
    report.append("")

    report.append("=" * 80)
    report.append("END OF REPORT")
    report.append("=" * 80)

    # Save report
    report_text = "\n".join(report)
    with open(f"{output_dir}/{DATASET_NAME}_{model_name}_experiment_summary.txt", 'w', encoding='utf-8') as f:
        f.write(report_text)

    return report_text


# ============================================================================
# MAIN EXECUTION
# ============================================================================
def main():
    global DATASET_NAME

    print("=" * 80)
    print("MULTI-DATASET MODEL COMPARISON V6 - SMART EARLY STOPPING")
    print("=" * 80)
    print(f"Data Directory: {DATA_DIR}")
    print(f"Output Directory: {OUTPUT_DIR}")
    print(f"Device: {DEVICE}")
    print("=" * 80)

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load and prepare data
    print("\n[1] LOADING AND PREPARING DATA...")
    data = load_and_prepare_data()

    # Auto-detect dataset name if not set
    if DATASET_NAME is None:
        DATASET_NAME = Path(DATA_DIR).name.replace('-', '_').lower()
    print(f"Dataset Name: {DATASET_NAME}")

    # Save preprocessing artifacts for inference
    preprocessing_path = f"{OUTPUT_DIR}/preprocessing"
    os.makedirs(preprocessing_path, exist_ok=True)
    joblib.dump(data['le'], f"{preprocessing_path}/{DATASET_NAME}_label_encoder.joblib")
    print(f"âœ“ Label encoder saved: {preprocessing_path}/{DATASET_NAME}_label_encoder.joblib")

    # Results storage
    all_results = {}

    # =========================================================================
    # BASELINE MODELS (No optimization)
    # =========================================================================
    if ENABLED_BASELINE_MODELS:
        print("\n" + "=" * 80)
        print("[2] TRAINING BASELINE MODELS (No Optimization)")
        print("=" * 80)

        baseline_models = {
            'Decision_Tree': (DecisionTreeClassifier, {'max_depth': 20, 'random_state': CONFIG['random_state']}),
            'KNN': (KNeighborsClassifier, {'n_neighbors': 5, 'n_jobs': -1}),
            'Naive_Bayes': (GaussianNB, {}),
            'Logistic_Regression': (LogisticRegression, {'max_iter': 1000, 'random_state': CONFIG['random_state'], 'n_jobs': -1}),
        }

        for model_name in ENABLED_BASELINE_MODELS:
            if model_name not in baseline_models:
                print(f"  âš  Unknown baseline model: {model_name}")
                continue

            print(f"\n  Training {model_name}...")
            model_class, params = baseline_models[model_name]

            start_time = time.time()
            model = model_class(**params)
            model.fit(data['X_train'], data['y_train'])
            y_pred = model.predict(data['X_test'])
            training_time = time.time() - start_time

            results = evaluate_model(data['y_test'], y_pred, data['le'])
            results['training_time'] = training_time

            # Check mode collapse
            is_collapsed, collapse_info = check_mode_collapse(y_pred, data['n_classes'])

            # Plot confusion matrix
            plot_confusion_matrix(
                np.array(results['confusion_matrix']),
                data['le'].classes_,
                model_name,
                get_model_output_dir(model_name)
            )

            # SHAP (for tree-based only)
            shap_importance = None
            shap_error = None
            if 'Decision_Tree' in model_name and SHAP_AVAILABLE:
                shap_importance, shap_error = compute_shap_explanations(
                    model, data['X_train'], data['X_test'],
                    data['feature_names'], model_name, get_model_output_dir(model_name)
                )
            elif SHAP_AVAILABLE:
                shap_error = "SHAP only computed for tree-based baseline models"

            # LIME
            lime_results = None
            lime_error = None
            if LIME_AVAILABLE:
                lime_results, lime_error = compute_lime_explanations(
                    model, data['X_train'], data['X_test'], data['y_test'],
                    data['feature_names'], data['le'], model_name, get_model_output_dir(model_name)
                )

            # Generate report
            generate_summary_report(
                model_name, results, params, training_time, get_model_output_dir(model_name),
                data, shap_importance, lime_results, collapse_info, shap_error, lime_error
            )

            # Save results
            with open(f"{get_model_output_dir(model_name)}/{DATASET_NAME}_{model_name}_results.json", 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2)
            with open(f"{get_model_output_dir(model_name)}/{DATASET_NAME}_{model_name}_best_params.json", 'w', encoding='utf-8') as f:
                json.dump(params, f, indent=2)

            # Save trained model
            model_path = f"{get_model_output_dir(model_name)}/{DATASET_NAME}_{model_name}_model.joblib"
            joblib.dump(model, model_path)
            print(f"  âœ“ Model saved: {model_path}")

            all_results[model_name] = results
            print(f"  âœ“ {model_name}: F1={results['f1_macro']:.4f}, Acc={results['accuracy']:.4f}, Time={training_time:.2f}s")

    # =========================================================================
    # OPTIMIZED ML MODELS
    # =========================================================================
    if ENABLED_ML_MODELS:
        print("\n" + "=" * 80)
        print("[3] TRAINING OPTIMIZED ML MODELS")
        print("=" * 80)

        ml_optimizers = {
            'XGBoost': optimize_xgboost,
            'LightGBM': optimize_lightgbm,
            'Random_Forest': optimize_random_forest,
            'SVM_RBF': optimize_svm,
            'CatBoost': optimize_catboost,
        }

        ml_model_classes = {
            'XGBoost': XGBClassifier,
            'LightGBM': lambda **p: __import__('lightgbm').LGBMClassifier(**p),
            'Random_Forest': RandomForestClassifier,
            'SVM_RBF': SVC,
            'CatBoost': lambda **p: __import__('catboost').CatBoostClassifier(**p),
        }

        for model_name in ENABLED_ML_MODELS:
            if model_name not in ml_optimizers:
                print(f"  âš  Unknown ML model: {model_name}")
                continue

            print(f"\n  Optimizing {model_name}...")
            start_time = time.time()

            # Optimize
            best_params, best_val_f1 = ml_optimizers[model_name](data, n_trials=CONFIG['optuna_trials_ml'])
            print(f"  Best Val F1: {best_val_f1:.4f}")

            # Train final model
            if model_name == 'XGBoost':
                use_gpu = best_params.pop('_use_gpu', False)
                best_params['random_state'] = CONFIG['random_state']
                best_params['verbosity'] = 0
                best_params['use_label_encoder'] = False
                if use_gpu:
                    best_params['tree_method'] = 'hist'
                    best_params['device'] = 'cuda'
                model = XGBClassifier(**best_params)
            elif model_name == 'LightGBM':
                from lightgbm import LGBMClassifier
                use_gpu = best_params.pop('_use_gpu', False)
                best_params['verbose'] = -1
                best_params['random_state'] = CONFIG['random_state']
                best_params['n_jobs'] = -1
                if use_gpu:
                    best_params['device'] = 'gpu'
                model = LGBMClassifier(**best_params)
            elif model_name == 'Random_Forest':
                best_params['random_state'] = CONFIG['random_state']
                best_params['n_jobs'] = -1
                model = RandomForestClassifier(**best_params)
            elif model_name == 'SVM_RBF':
                model = SVC(**best_params)
            elif model_name == 'CatBoost':
                from catboost import CatBoostClassifier
                use_gpu = best_params.pop('_use_gpu', False)
                best_params['random_seed'] = CONFIG['random_state']
                best_params['verbose'] = False
                best_params['task_type'] = 'GPU' if use_gpu else 'CPU'
                best_params['auto_class_weights'] = 'Balanced'
                model = CatBoostClassifier(**best_params)

            model.fit(data['X_train'], data['y_train'])
            y_pred = model.predict(data['X_test'])
            training_time = time.time() - start_time

            results = evaluate_model(data['y_test'], y_pred, data['le'])
            results['training_time'] = training_time

            is_collapsed, collapse_info = check_mode_collapse(y_pred, data['n_classes'])

            # Confusion matrix
            plot_confusion_matrix(
                np.array(results['confusion_matrix']),
                data['le'].classes_,
                model_name,
                get_model_output_dir(model_name)
            )

            # SHAP
            shap_importance = None
            shap_error = None
            if SHAP_AVAILABLE:
                shap_importance, shap_error = compute_shap_explanations(
                    model, data['X_train'], data['X_test'],
                    data['feature_names'], model_name, get_model_output_dir(model_name)
                )

            # LIME
            lime_results = None
            lime_error = None
            if LIME_AVAILABLE:
                lime_results, lime_error = compute_lime_explanations(
                    model, data['X_train'], data['X_test'], data['y_test'],
                    data['feature_names'], data['le'], model_name, get_model_output_dir(model_name)
                )

            # Report
            generate_summary_report(
                model_name, results, best_params, training_time, get_model_output_dir(model_name),
                data, shap_importance, lime_results, collapse_info, shap_error, lime_error
            )

            # Save
            with open(f"{get_model_output_dir(model_name)}/{DATASET_NAME}_{model_name}_results.json", 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2)
            with open(f"{get_model_output_dir(model_name)}/{DATASET_NAME}_{model_name}_best_params.json", 'w', encoding='utf-8') as f:
                json.dump(best_params, f, indent=2)

            # Save trained model
            model_path = f"{get_model_output_dir(model_name)}/{DATASET_NAME}_{model_name}_model.joblib"
            joblib.dump(model, model_path)
            print(f"  âœ“ Model saved: {model_path}")

            all_results[model_name] = results
            print(f"  âœ“ {model_name}: F1={results['f1_macro']:.4f}, Acc={results['accuracy']:.4f}, Time={training_time:.2f}s")

    # =========================================================================
    # DEEP LEARNING MODELS
    # =========================================================================
    if ENABLED_DL_MODELS and TORCH_AVAILABLE:
        print("\n" + "=" * 80)
        print("[4] TRAINING DEEP LEARNING MODELS (Mode Collapse Fixed)")
        print("=" * 80)

        dl_models = {
            'MLP': ImprovedMLP,
            'CNN_1D': Improved1DCNN,
            'LSTM': ImprovedLSTM,
            'GRU': ImprovedGRU,
            'CNN_LSTM': ImprovedCNN_LSTM,
            'Transformer': ImprovedTransformer,
        }

        for model_name in ENABLED_DL_MODELS:
            if model_name not in dl_models:
                print(f"  âš  Unknown DL model: {model_name}")
                continue

            print(f"\n  Training {model_name}...")
            model_class = dl_models[model_name]
            start_time = time.time()

            # Optimize with mode collapse prevention
            best_params, best_val_f1 = optimize_dl_model(
                model_class, model_name, data,
                n_trials=CONFIG['optuna_trials_dl']
            )
            print(f"  Best Val F1: {best_val_f1:.4f}")

            # Skip if no learning occurred
            if best_val_f1 < 0.10:
                print(f"  âš  SKIPPING {model_name}: Model failed to learn (F1 < 10%)")
                print(f"    This is expected for MLP/CNN on tabular data (see Grinsztajn et al., NeurIPS 2022)")

                # Save minimal results
                results = {
                    'accuracy': 0.0,
                    'f1_macro': best_val_f1,
                    'status': 'SKIPPED - Mode Collapse',
                    'reason': 'Model unsuitable for tabular data'
                }
                all_results[model_name] = results

                with open(f"{get_model_output_dir(model_name)}/{DATASET_NAME}_{model_name}_results.json", 'w', encoding='utf-8') as f:
                    json.dump(results, f, indent=2)

                continue  # Skip to next model

            # Build final model with best params
            if model_name == 'MLP':
                hidden_dim = best_params.get('hidden_dim', 256)
                num_layers = best_params.get('num_layers', 2)
                hidden_dims = [hidden_dim // (2**i) for i in range(num_layers)]
                hidden_dims = [max(32, h) for h in hidden_dims]
                final_model = model_class(
                    input_dim=data['n_features'],
                    num_classes=data['n_classes'],
                    hidden_dims=hidden_dims,
                    dropout=best_params.get('dropout', 0.3)
                )
            elif model_name == 'CNN_1D':
                final_model = model_class(
                    input_dim=data['n_features'],
                    num_classes=data['n_classes'],
                    num_filters=best_params.get('num_filters', 64),
                    dropout=best_params.get('dropout', 0.3)
                )

            elif model_name == 'LSTM':
                final_model = model_class(
                    input_dim=data['n_features'],
                    num_classes=data['n_classes'],
                    hidden_size=best_params.get('hidden_size', 256),
                    num_layers=best_params.get('num_layers', 2),
                    dropout=best_params.get('dropout', 0.3)
                )

            elif model_name == 'GRU':
                final_model = model_class(
                    input_dim=data['n_features'],
                    num_classes=data['n_classes'],
                    hidden_size=best_params.get('hidden_size', 256),
                    num_layers=best_params.get('num_layers', 2),
                    dropout=best_params.get('dropout', 0.3)
                )

            elif model_name == 'CNN_LSTM':
                final_model = model_class(
                    input_dim=data['n_features'],
                    num_classes=data['n_classes'],
                    num_filters=best_params.get('num_filters', 64),
                    hidden_size=best_params.get('hidden_size', 128),
                    dropout=best_params.get('dropout', 0.3)
                )

            elif model_name == 'Transformer':
                d_model = best_params.get('d_model', 64)
                nhead = best_params.get('nhead', 4)
                # Ensure d_model is divisible by nhead
                if d_model % nhead != 0:
                    d_model = (d_model // nhead) * nhead
                final_model = model_class(
                    input_dim=data['n_features'],
                    num_classes=data['n_classes'],
                    d_model=d_model,
                    nhead=nhead,
                    num_layers=best_params.get('num_layers', 2),
                    dropout=best_params.get('dropout', 0.3)
                )

            # Train final model with more epochs
            final_model, _ = train_dl_model(
                final_model, data,
                epochs=100,
                lr=best_params.get('lr', 1e-3),
                batch_size=best_params.get('batch_size', 512),
                patience=20,
                weight_decay=best_params.get('weight_decay', 1e-4),
                use_focal_loss=True
            )
            training_time = time.time() - start_time

            # Predict
            final_model.eval()
            with torch.no_grad():
                X_test_tensor = torch.FloatTensor(data['X_test']).to(DEVICE)
                outputs = final_model(X_test_tensor)
                y_pred = outputs.argmax(dim=1).cpu().numpy()

            results = evaluate_model(data['y_test'], y_pred, data['le'])
            results['training_time'] = training_time

            is_collapsed, collapse_info = check_mode_collapse(y_pred, data['n_classes'])
            if is_collapsed:
                print(f"  âš  Mode collapse detected! Dominant class ratio: {collapse_info['dominant_ratio']:.2%}")

            # Confusion matrix
            plot_confusion_matrix(
                np.array(results['confusion_matrix']),
                data['le'].classes_,
                model_name,
                get_model_output_dir(model_name)
            )

            # LIME only (SHAP not supported for DL)
            lime_results = None
            lime_error = None
            shap_error = "SHAP not supported for Deep Learning models"
            if LIME_AVAILABLE:
                # Create wrapper for LIME
                def predict_fn(X):
                    final_model.eval()
                    with torch.no_grad():
                        X_tensor = torch.FloatTensor(X).to(DEVICE)
                        outputs = final_model(X_tensor)
                        probs = torch.softmax(outputs, dim=1).cpu().numpy()
                    return probs

                class ModelWrapper:
                    def __init__(self, fn):
                        self.predict_proba = fn
                    def predict(self, X):
                        return self.predict_proba(X).argmax(axis=1)

                wrapper = ModelWrapper(predict_fn)
                lime_results, lime_error = compute_lime_explanations(
                    wrapper, data['X_train'], data['X_test'], data['y_test'],
                    data['feature_names'], data['le'], model_name, get_model_output_dir(model_name)
                )

            # Report
            generate_summary_report(
                model_name, results, best_params, training_time, get_model_output_dir(model_name),
                data, None, lime_results, collapse_info, shap_error, lime_error
            )

            # Save
            with open(f"{get_model_output_dir(model_name)}/{DATASET_NAME}_{model_name}_results.json", 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2)
            with open(f"{get_model_output_dir(model_name)}/{DATASET_NAME}_{model_name}_best_params.json", 'w', encoding='utf-8') as f:
                json.dump(best_params, f, indent=2)

            # Save trained PyTorch model
            model_path = f"{get_model_output_dir(model_name)}/{DATASET_NAME}_{model_name}_model.pt"
            torch.save({
                'model_state_dict': final_model.state_dict(),
                'model_class': model_name,
                'input_dim': data['n_features'],
                'num_classes': data['n_classes'],
                'best_params': best_params
            }, model_path)
            print(f"  âœ“ Model saved: {model_path}")

            all_results[model_name] = results
            print(f"  âœ“ {model_name}: F1={results['f1_macro']:.4f}, Acc={results['accuracy']:.4f}, Time={training_time:.2f}s")

    # =========================================================================
    # FINAL SUMMARY
    # =========================================================================
    print("\n" + "=" * 80)
    print("FINAL RESULTS SUMMARY")
    print("=" * 80)
    print(f"\n{'Model':<20} {'Accuracy':>10} {'F1 (Macro)':>12} {'Precision':>10} {'Recall':>10} {'MCC':>10} {'Time (s)':>10}")
    print("-" * 90)

    for model_name, results in sorted(all_results.items(), key=lambda x: x[1]['f1_macro'], reverse=True):
        print(f"{model_name:<20} {results['accuracy']:>10.4f} {results['f1_macro']:>12.4f} "
              f"{results['precision_macro']:>10.4f} {results['recall_macro']:>10.4f} "
              f"{results['mcc']:>10.4f} {results.get('training_time', 0):>10.1f}")

    # Save overall summary
    summary = {
        'dataset': DATASET_NAME,
        'timestamp': datetime.now().isoformat(),
        'models': {name: {
            'accuracy': r['accuracy'],
            'f1_macro': r['f1_macro'],
            'f1_weighted': r['f1_weighted'],
            'precision_macro': r['precision_macro'],
            'recall_macro': r['recall_macro'],
            'mcc': r['mcc'],
            'training_time': r.get('training_time', 0)
        } for name, r in all_results.items()}
    }

    with open(f"{OUTPUT_DIR}/{DATASET_NAME}_overall_summary.json", 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)

    print(f"\nâœ“ All results saved to: {OUTPUT_DIR}/")
    print("=" * 80)


if __name__ == "__main__":
    main()