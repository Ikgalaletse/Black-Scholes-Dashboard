# 📈 Black–Scholes Option Pricer Dashboard

This repository contains an **interactive Streamlit dashboard** that implements the **Black–Scholes option pricing model** for European options.

The dashboard is designed to **explain, visualise, and analyse option pricing behaviour** by allowing users to interactively change model parameters and immediately observe their impact on option prices, sensitivities (Greeks), and probabilities.

---

## 📌 What the Dashboard Does

The dashboard provides a **hands-on interface** for understanding the Black–Scholes model. It allows users to:

- Price European **Call and Put options**
- Analyse **option sensitivities (Greeks)**
- Visualise **price sensitivity to spot price and volatility**
- Explore **risk-neutral probabilities** of expiring in or out of the money

All outputs update **in real time** as inputs are adjusted.

---

## 🧮 Core Dashboard Components

### 1️⃣ Option Pricing

The dashboard calculates **Call and Put prices** using the Black–Scholes formula based on the following inputs:

- Spot price ($S$)
- Strike price ($K$)
- Time to maturity ($T$)
- Volatility ($\sigma$)
- Risk-free interest rate ($r$)

It also displays the intermediate variables:

- $d_1$
- $d_2$

This provides transparency into how option prices are derived.

---

### 2️⃣ Greeks (Sensitivity Analysis)

The **Greeks tab** shows how option prices respond to small changes in model parameters:

- **Delta** – sensitivity to changes in spot price  
- **Gamma** – curvature with respect to spot price  
- **Vega** – sensitivity to volatility  
- **Theta** – sensitivity to time decay  
- **Rho** – sensitivity to interest rates  

Both **Call and Put Greeks** are displayed side-by-side in a clean table.

> Vega is reported per **1.00 change in volatility**, consistent with quantitative finance conventions.

---

### 3️⃣ Price Sensitivity Heatmaps

The **Heatmaps tab** visualises how option prices change across a grid of:

- Spot prices
- Volatility levels

Features include:
- Separate heatmaps for **Call** and **Put** options
- Adjustable parameter ranges
- Configurable grid resolution
- Reduced visual clutter for clarity

These heatmaps highlight the **non-linear relationship** between option prices, spot price, and volatility.

---

### 4️⃣ Risk-Neutral Probability Analysis

The **Probabilities tab** shows the **risk-neutral probability** that an option expires:

- **In-the-Money (ITM)**
- **Out-of-the-Money (OTM)**

These probabilities are computed using **\( N(d_2) \)** from the Black–Scholes framework and are presented as:

- A probability table across spot prices
- Line plots for intuitive interpretation

> ⚠️ These are **risk-neutral probabilities**, not real-world or historical probabilities.

---

## 🎛 Dashboard Design Principles

The dashboard follows **best practices for analytical dashboards**:

- Clear hierarchy (KPIs → Tabs → Details)
- Minimal visual clutter
- Logical grouping of inputs in the sidebar
- Expanders used to hide secondary information
- Cached computations for better performance

---

## 🧠 Model Assumptions

- European options only  
- No dividends  
- Constant volatility and interest rate  
- Frictionless markets  
- Risk-neutral valuation  

---

## 🔧 Technologies & Dependencies

- **Python**
- **Streamlit** – interactive dashboard framework
- **NumPy** – numerical computation
- **Pandas** – data handling
- **SciPy** – normal distribution functions
- **Matplotlib** – lightweight visualisation

Install dependencies:
```bash
pip install streamlit numpy pandas scipy matplotlib

---

▶️ Run the Dashboard Locally

Clone the repository, install the dependencies, and run:

streamlit run app.py

---

👤 Author

Ikgalaletse Keatlegile Neo Sebola
🔗 LinkedIn: https://www.linkedin.com/in/neo-sebola-499b72313/



