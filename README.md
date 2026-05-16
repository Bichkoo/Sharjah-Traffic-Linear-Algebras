# 🚘 Sharjah Commute Optimizer

A data-driven web app that predicts Sharjah traffic congestion and calculates your **mathematically optimal departure time** — built from scratch using pure linear algebra, without any ML libraries.

![Python](https://img.shields.io/badge/Python-3.8+-blue?style=flat-square&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red?style=flat-square&logo=streamlit)
![NumPy](https://img.shields.io/badge/NumPy-OLS-orange?style=flat-square&logo=numpy)
![Plotly](https://img.shields.io/badge/Plotly-Interactive-green?style=flat-square&logo=plotly)

---

## 📌 What It Does

You enter:
- Day of the week
- Target arrival time
- Route distance (km)

The app returns the **latest safe departure time** and the **minimum drive time window**, backed by a traffic prediction model trained on real Sharjah congestion data.

---

## 🧮 The Math Behind It

This project was built as a direct application of linear algebra concepts — no `sklearn`, no `statsmodels`. Everything is implemented from first principles.

### 1. Ordinary Least Squares (OLS) via Matrix Algebra

Given hundreds of hourly congestion observations, the system `Xβ = y` is overdetermined (more equations than unknowns). The closed-form solution is derived by solving the **Normal Equations**:

$$\beta = (X^T X)^{-1} X^T y$$

Implemented in NumPy as:
```python
beta = np.linalg.inv(X.T @ X) @ X.T @ y
```

### 2. Fourier Feature Engineering

Traffic follows a cyclical 24-hour pattern. To capture this, raw hour values are mapped to **Fourier harmonics**:

| Feature | Formula | Captures |
|---|---|---|
| `sin24`, `cos24` | `sin/cos(2πt/24)` | Daily cycle |
| `sin12`, `cos12` | `sin/cos(4πt/24)` | Twice-daily peaks (AM/PM rush) |
| `sin8`, `cos8` | `sin/cos(6πt/24)` | 8-hour sub-patterns |

### 3. Workday Interaction Terms

A binary workday indicator `W = 1` (Mon–Fri) or `W = 0` (Sat–Sun) is multiplied with the Fourier features to model how rush hour patterns differ between weekdays and weekends.

### 4. Constrained Optimization

The departure time `t_dep` is optimized under the constraint:

$$t_{dep} + \frac{D(t_{dep})}{60} \leq t_{target}$$

Where `D(t)` is the predicted drive time accounting for congestion-induced delay.

---

## 🚀 Getting Started

### Prerequisites
```bash
pip install -r requirements.txt
```

### Run the App
```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`

---

## 📁 Project Structure

```
Sharjah-Traffic-Linear-Algebras/
│
├── app.py                        # Main Streamlit app + math engine
├── least_squares_analysis.ipynb  # Exploratory analysis & model development
├── sharjah_congestion.csv        # Hourly congestion dataset (Mon–Sun)
└── requirements.txt              # Python dependencies
```

---

## 📊 Dataset

`sharjah_congestion.csv` contains hourly (0–23) traffic congestion percentages across all 7 days of the week, sourced from Sharjah road network observations.

---

## 🛠 Tech Stack

- **Streamlit** — interactive web dashboard
- **NumPy** — matrix operations and OLS solver
- **Pandas** — data wrangling
- **Plotly** — interactive traffic curve visualization

---

## 👤 Author

**Bichkoo** — Freshman, Applied Mathematics, Khalifa University  
[GitHub](https://github.com/Bichkoo)
