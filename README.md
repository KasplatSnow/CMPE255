# CMPE 255 — Keystroke Dynamics User Classification

This project performs biometric user identification based on keystroke dynamics from the [DSL-StrongPassword dataset](https://www.cs.cmu.edu/~keystroke/).

We use models like **XGBoost**, **Random Forest**, and **MLP** to classify users by their typing patterns on a fixed password.

---

## 📦 Project Structure

```
├── data/
│   └── DSL-StrongPasswordData.csv       # Raw dataset
├── notebooks/
│   └── final_results.ipynb              # Visualizations & evaluation
├── results/
│   ├── metrics.csv                      # Accuracy/F1/AUC logs
│   └── figures/
│       ├── confmat_xgb.png              # Confusion matrix plot
│       └── confmat_rf.png
├── src/
│   ├── train.py                         # Main training script
│   ├── models.py                        # Model factory
│   ├── preprocessing.py                 # Data pipeline (clipping, scaling, filtering)
│   └── utils.py                         # Logging, seeding, metric saving
└── README.md
```

---

## 🚀 Setup Instructions

```bash
python -m venv venv
source venv/bin/activate       # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

---

## 🔁 Run the Pipeline

Train on the top-10 most frequent users using XGBoost:

```bash
python src/train.py --model xgb
```

Other model choices: `rf`, `logreg`, `mlp`, `knn`

---

## 📊 Outputs

- `results/metrics.csv` – accuracy, f1, and AUC per model run
- `results/figures/confmat_<model>.png` – confusion matrix
- Sample metric:
  ```
  {'model': 'xgb', 'acc': 0.932, 'f1': 0.932, 'auc': 0.9965}
  ```

---

## ⚙️ Preprocessing

Each repetition of the password is converted into 91 features:
- 31 key hold times (`H.`)
- 30 digraph down-down latencies (`DD.`)
- 30 up-down transitions (`UD.`)

We apply:
- Outlier clipping at the 99th percentile
- Standard scaling
- Low variance filtering (threshold = 0.0005)

---

## 🧠 Model Summary

| Model       | Notes                                          |
|-------------|-------------------------------------------------|
| MLP         | Best performer (93.6% accuracy, best F1 & AUC)  |
| XGBoost     | Strong generalization (93.2% accuracy)          |
| KNN         | Competitive baseline (91.7% accuracy)           |
| RandomForest| Fast, interpretable, slightly lower AUC         |
| LogReg      | Simple linear baseline                          |

---

## 📚 Acknowledgments

Dataset: [CMU DSL Strong Password Dataset](https://www.cs.cmu.edu/~keystroke/)
