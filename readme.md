# 🌿 ETo Prediction System
### ANN-based Reference Evapotranspiration Estimation using Deep Learning

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://eto-predictor-new.streamlit.app)
[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange?logo=pytorch)](https://pytorch.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-red?logo=streamlit)](https://streamlit.io)
[![License](https://img.shields.io/badge/License-Academic-green)](LICENSE)

---

## 📖 Overview

The **ETo Prediction System** is an interactive web application for estimating **Reference Evapotranspiration (ETo)** using Artificial Neural Networks (ANN). The models are trained on **50 years of daily climate data (1970–2020)** and cover all possible combinations of 6 climate variables.

ETo is a critical parameter in hydrology, agriculture, and water resource management. It represents the evapotranspiration from a hypothetical reference crop (short grass) and is widely used for irrigation scheduling and water balance studies.

> 🔬 **Only models with R² ≥ 0.90 on the test set are available** — ensuring every prediction is backed by a high-accuracy, validated model.

---

## ✨ Features

### 🤖 Smart Auto Model Selection
- Check only the parameters you have data for
- App automatically finds the **best matching model** (highest R²) for your exact parameter combination
- If no model exists, it tells you **exactly which parameter to add or remove** to unlock a valid model
- **No fallback to lower quality models** — strict R² ≥ 0.90 enforcement

### ✅ Full Input Validation
- Range validation for all 6 parameters
- Cross-validation: Tmax must be > Tmin
- Cross-validation: RHmax must be > RHmin
- Predict button stays **disabled** until all inputs are valid
- Warnings for unrealistic ETo predictions (< 0 or > 15 mm/day)

### 📊 Model Rankings Tab
- Color-coded performance table for all 24 qualified models
- Filter by number of input parameters (2–6)
- View R², RMSE, MAE, NSE for Train and Test sets

### ℹ️ Parameter Guide Tab
- Valid ranges and descriptions for all 6 parameters
- Common input errors and how to fix them
- Number of qualifying models each parameter appears in

### 🎨 UI/UX
- Fully **dark mode compatible** design
- Responsive 2-column layout
- Parameter badges showing which inputs each model uses
- Clean result card displaying predicted ETo in mm/day

---

## 🧠 Model Architecture

```
Input Layer (2–6 features)
        ↓
Linear → BatchNorm1d → ELU
        ↓
Linear → BatchNorm1d → ELU → Dropout
        ↓
Linear → BatchNorm1d → ELU → Dropout
        ↓
Linear → BatchNorm1d → ELU → Dropout
        ↓
Linear → BatchNorm1d → ELU
        ↓
Output Layer (1 neuron → ETo in mm/day)
```

| Property | Value |
|---|---|
| Architecture | 5-Layer Deep ANN |
| Activation Function | ELU (Exponential Linear Unit) |
| Regularization | BatchNorm1d + Dropout |
| Optimizer | Adam |
| Loss Function | MSE |
| Framework | PyTorch |
| Training Data | 1970–2020 (50 years daily) |
| Total Combinations Trained | 57 models |
| Qualified Models (R² ≥ 0.90) | 24 models |

---

## 🌦️ Input Parameters

| # | Parameter | Symbol | Unit | Valid Range | Description |
|---|---|---|---|---|---|
| 1 | Sunshine Hours | n | hrs/day | 0.0 – 14.0 | Daily sunshine duration |
| 2 | Max Temperature | Tmax | °C | -10.0 – 50.0 | Daily maximum air temperature |
| 3 | Min Temperature | Tmin | °C | -20.0 – 40.0 | Daily minimum air temperature |
| 4 | Max Relative Humidity | RHmax | % | 10.0 – 100.0 | Daily maximum relative humidity |
| 5 | Min Relative Humidity | RHmin | % | 5.0 – 100.0 | Daily minimum relative humidity |
| 6 | Wind Speed | u | m/s | 0.0 – 10.0 | Average daily wind speed at 2m height |

> ⚠️ **Minimum 2 parameters are required.** The app supports all 24 valid combinations.

---

## 📊 Top 10 Models

| Rank | Model ID | Input Features | Inputs | R² Test | RMSE | MAE | NSE |
|---|---|---|---|---|---|---|---|
| 1 | M57 | n + Tmax + Tmin + RHmax + RHmin + u | 6 | 0.9805 | 0.2840 | 0.2089 | 0.9805 |
| 2 | M53 | n + Tmax + Tmin + RHmin + u | 5 | 0.9785 | 0.2986 | 0.2182 | 0.9785 |
| 3 | M54 | n + Tmax + RHmax + RHmin + u | 5 | 0.9771 | 0.3082 | 0.2252 | 0.9771 |
| 4 | M52 | n + Tmax + Tmin + RHmax + u | 5 | 0.9770 | 0.3090 | 0.2279 | 0.9770 |
| 5 | M41 | n + Tmax + RHmin + u | 4 | 0.9739 | 0.3287 | 0.2432 | 0.9739 |
| 6 | M38 | n + Tmax + Tmin + u | 4 | 0.9715 | 0.3437 | 0.2544 | 0.9715 |
| 7 | M55 | n + Tmin + RHmax + RHmin + u | 5 | 0.9713 | 0.3451 | 0.2532 | 0.9713 |
| 8 | M44 | n + Tmin + RHmin + u | 4 | 0.9675 | 0.3670 | 0.2666 | 0.9675 |
| 9 | M40 | n + Tmax + RHmax + u | 4 | 0.9669 | 0.3707 | 0.2762 | 0.9669 |
| 10 | M19 | n + Tmax + u | 3 | 0.9611 | 0.4014 | 0.3064 | 0.9611 |

---

## 🔢 Parameter Combinations Summary

| Number of Inputs | Total Combinations | Qualified (R² ≥ 0.90) |
|---|---|---|
| 2 inputs | 15 | varies |
| 3 inputs | 20 | varies |
| 4 inputs | 15 | varies |
| 5 inputs | 6 | varies |
| 6 inputs | 1 | 1 (Best: R²=0.9805) |
| **Total** | **57** | **24** |

---

## 🚀 Getting Started

### Prerequisites
- Python 3.9 or higher
- pip

### 1. Clone the Repository
```bash
git clone https://github.com/ayushijhagsm-tech/eto-prediction.git
cd eto-prediction
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Verify Folder Structure
Make sure your folder looks like this:
```
eto-prediction/
├── app.py
├── requirements.txt
├── README.md
├── model_scores_pt.csv
├── ann_models_pt/
│   ├── M57_n_Tmax_Tmin_RHmax_RHmin_u_full.pt
│   ├── M53_n_Tmax_Tmin_RHmin_u_full.pt
│   └── ... (24 .pt files total)
└── ann_scalers_pt/
    ├── M57_n_Tmax_Tmin_RHmax_RHmin_u_scaler_X.pkl
    ├── M57_n_Tmax_Tmin_RHmax_RHmin_u_scaler_y.pkl
    └── ... (48 .pkl files total)
```

### 4. Run the App
```bash
streamlit run app.py
```

Open your browser at **http://localhost:8501** 🎉

---

## ☁️ Deployment

### Option 1 — Streamlit Cloud (Recommended — Free)

1. Push your repository to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Click **New App**
4. Select your GitHub repository
5. Set **Main file path** → `app.py`
6. Click **Deploy** ✅

Your app will be live at:
`https://YOUR_USERNAME-eto-prediction-app-XXXXX.streamlit.app`

> **⚠️ Large Files?** If your `.pt` or `.pkl` files exceed GitHub's 100MB limit, use Git LFS:
> ```bash
> git lfs install
> git lfs track "*.pt"
> git lfs track "*.pkl"
> git add .gitattributes
> git add .
> git commit -m "Track large files with LFS"
> git push
> ```

### Option 2 — Hugging Face Spaces (No GitHub Required)

1. Go to [huggingface.co/spaces](https://huggingface.co/spaces)
2. Click **Create new Space**
3. Choose **Streamlit** as the SDK
4. Upload your files directly via the web interface
5. Add a `requirements.txt` — app goes live automatically ✅

---

## 📦 Requirements

```
streamlit
torch
pandas
scikit-learn
numpy
openpyxl
```

Install all:
```bash
pip install -r requirements.txt
```

---

## ⚠️ Common Errors & Fixes

| Error | Cause | Fix |
|---|---|---|
| `No model found` | Parameter combo not in 24 qualified models | Add or remove a parameter as suggested |
| `Tmax ≤ Tmin` | Max temp must exceed min | Increase Tmax or decrease Tmin |
| `RHmax ≤ RHmin` | Max humidity must exceed min | Increase RHmax or decrease RHmin |
| `Sunshine > 14 hrs` | Physically impossible | Max sunshine is 14 hrs/day |
| `Wind speed < 0` | Cannot be negative | Enter value ≥ 0 m/s |
| `RH > 100%` | Cannot exceed 100% | Enter value between 5–100% |
| `ETo < 0 mm/day` | Input combination unusual | Verify all values are realistic |
| `ETo > 15 mm/day` | Unusually high | Check Tmax and wind speed |
| `Missing model files` | `.pt` or `.pkl` files not in correct folder | Ensure `ann_models_pt/` and `ann_scalers_pt/` folders exist |
| `model_scores_pt.csv not found` | CSV not in root folder | Place CSV in same folder as `app.py` |

---

## 📂 File Naming Convention

Model files follow this naming pattern:

```
{Model_ID}_full.pt               → PyTorch model weights
{Model_ID}_scaler_X.pkl          → Input feature scaler
{Model_ID}_scaler_y.pkl          → Output (ETo) scaler
```

Example for the best model (M57):
```
ann_models_pt/M57_n_Tmax_Tmin_RHmax_RHmin_u_full.pt
ann_scalers_pt/M57_n_Tmax_Tmin_RHmax_RHmin_u_scaler_X.pkl
ann_scalers_pt/M57_n_Tmax_Tmin_RHmax_RHmin_u_scaler_y.pkl
```

---

## 🔬 Technical Details

### Why ETo?
Reference Evapotranspiration (ETo) is used to:
- Schedule irrigation for crops
- Estimate water demand for a region
- Study effects of climate change on water availability
- Calculate actual crop water requirements (ETc = ETo × Kc)

### Why ANN?
Artificial Neural Networks capture **non-linear relationships** between climate variables that traditional empirical equations (like Penman-Monteith) may miss when data is limited. The models here are trained to **approximate FAO-56 Penman-Monteith** results using fewer input variables.

### Performance Metrics Used

| Metric | Full Form | Best Value |
|---|---|---|
| R² | Coefficient of Determination | 1.0 |
| RMSE | Root Mean Square Error | 0.0 |
| MAE | Mean Absolute Error | 0.0 |
| NSE | Nash-Sutcliffe Efficiency | 1.0 |
| MBE | Mean Bias Error | 0.0 |

---

## 👨‍💻 Author

**ayushi jha**
- GitHub: [@ayushijha](https://github.com/ayushijhagsm-tech)

---

## 📄 License

This project is developed for **academic and research purposes**.
For commercial use, please contact the author.

---

## 🙏 Acknowledgements

- **FAO-56 Penman-Monteith** method used as reference standard for ETo
- **PyTorch** for model training and inference
- **Streamlit** for the interactive web interface
- Climate data spanning **1970–2020** used for training and validation
