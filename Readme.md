# GLoVE: GARCH-Informed Deep Learning for Volatility Forecasting

[中文版本](readme-chinese.md) | [Reproduction Guide](reproduction.md)

![Header Image](https://img.shields.io/badge/Finance-Data%20Science-blue) ![AIEnhanced](https://img.shields.io/badge/GARCH-Informed-gold) ![Taiwan50](https://img.shields.io/badge/Dataset-Taiwan%2050-green)

> **"Bridging Econometric Stability with Deep Learning Scalability"**

Welcome to **GLoVE** (GARCH-informed Loss for Volatility Estimation), a project dedicated to advancing the state of multi-asset volatility forecasting. Our research explores the synthesis of classical econometric models (GARCH) and state-of-the-art neural network architectures (TSMixer) to provide more robust and accurate risk assessments in financial markets.

---

## 🏛️ Our Mission & Perspective

In the intersection of **Finance** and **Data Science**, we often face a trade-off: classical models like GARCH provide theoretical stability and tail robustness but struggle with high-dimensional nonlinear dynamics; conversely, deep learning models capture complex patterns but can be unstable or lack interpretability in financial contexts.

Our team has conducted a deep-dive investigation into **GARCH-informed deep learning**. By embedding GARCH reference forecasts directly into the training objective (mixed loss function), we enable neural networks to inherit the stability of econometrics while exploiting the predictive power of modern architectures.

---

## 🔬 Core Research & Contributions

This project reproduces and extends the foundational work of **Xu et al. [31]**, applying it to a modern, large-scale multi-asset setting (Taiwan 50 constituents, 2005–2025).

### 1. From Sequential to All-MLP Architecture
We transitioned from traditional recurrent architectures (LSTM) to the **TSMixer** (Time-Series Mixer), an all-MLP architecture. This allows for better exploitation of cross-sectional dependence across stocks without the computational overhead or vanishing gradient issues of RNNs.

### 2. The Role of the Mixing Parameter ($\lambda$)
A central finding of our research is that $\lambda$ is not just a tuning constant, but a **strategic steering wheel**. 
- **Higher $\lambda$**: Pulls the model toward GARCH stability, enhancing tail-sensitive metrics.
- **Lower $\lambda$**: Allows the model to prioritize average forecast accuracy through nonlinear representation learning.
We demonstrate that a carefully chosen $\lambda$ allows TSMixer to consistently outperform standalone GARCH models.

### 3. Shared-Parameter Optimization
By using a **shared-parameter design**, our model pools information across multiple assets. This mitigates idiosyncratic noise while preserving asset-specific dynamics through the GARCH-informed loss, making it highly effective for joint volatility prediction.

### 4. Critical View on Alpha Factors
Our empirical evidence suggests that high-dimensional **alpha signals** should be treated with caution. Without explicit screening or structural constraints, naive inclusion of these factors can introduce noise that overwhelms volatility dynamics. This reinforces the importance of embedding econometric structure (like GARCH) rather than relying solely on raw feature expansion.

---

## 📈 Experimental Context
- **Assets**: Taiwan 50 constituents.
- **Timeline**: 2005 - 2025 (Long-horizon evaluation covering multiple market cycles).
- **Objective**: Multi-step volatility forecasting with GARCH-informed supervision.

---

## 🏗️ Getting Started

For detailed instructions on how to set up the environment, run the preprocessing pipeline, and reproduce our experimental results, please refer to:

👉 **[Reproduction Guide (reproduction.md)](file:///Users/blackwingedkite/Desktop/GLoVE/reproduction.md)**

---

## 👥 Our Team

We are a group of researchers from **National Taiwan University (NTU)** committed to pushing the boundaries of financial AI:

- **Yu-Chieh Tsai** (r13723046@ntu.edu.tw)
- **Yu-Chi Ko** (r14946005@ntu.edu.tw)
- **Ta-Kang Kao** (r12922156@ntu.edu.tw)
- **Ho-Chien Huang** (r14946007@ntu.edu.tw)
- **Jovytta E. Yauwanta** (b12902087@csie.ntu.edu.tw)
- **Yu-Chen Kuo** (b12902061@csie.ntu.edu.tw)

---

## 🔗 Resources
- 📽️ **[Intro Video](https://www.youtube.com/watch?v=AhvbGsd1O98)**
- 📊 **[Slide Deck](https://docs.google.com/presentation/d/1RK-kOrt1dCIzXsmUMu1cpMOTFbq2gh3wWMnMSdsICV0/edit?usp=sharing)**
- 📁 **[Trained Models & Results](https://drive.google.com/drive/u/0/folders/1xbHQe9j2vbTLrdpKLcS7qy-QH8qNLHXI)**
