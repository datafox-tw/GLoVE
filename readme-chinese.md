# GLoVE: 基於 GARCH-Loss 的波動度預測實驗

![標語](https://img.shields.io/badge/金融-資料科學-blue) ![AI增強](https://img.shields.io/badge/GARCH-Informed-gold) ![台灣50](https://img.shields.io/badge/數據集-台灣50-green)

> **「在計量經濟學的穩定性與深度學習的可擴展性之間架起橋樑」**

[English Version](Readme.md) | [重現指南 (Reproduction Guide)](reproduction.md)

歡迎來到 **GLoVE** (GARCH-informed Loss for Volatility Estimation) 專案。本研究致力於推進多資產波動度預測的前沿技術，探索如何將傳統計量經濟模型 (GARCH) 與尖端神經網絡架構 (TSMixer) 結合，為金融市場提供更穩健且精確的風險評估。

---

## 🏛️ 我們的使命與觀點

在 **金融 (Finance)** 與 **資料科學 (Data Science)** 的交匯點上，我們經常面臨權衡：GARCH 等傳統模型提供了理論穩定性和對尾部風險的魯棒性，但在處理高維非線性動態時顯得力不從心；相反地，深度學習模型雖能捕捉複雜模式，但在金融脈絡下往往不夠穩定或缺乏解釋性。

我們團隊對 **「GARCH-informed 深度學習」** 進行了深入研究。透過將 GARCH 參考預測直接嵌入到訓練目標（混合損失函數）中，我們使神經網絡在繼承計量經濟學穩定性的同時，能充分發揮現代架構的預測能力。

---

## 🔬 核心研究與貢獻

本專案復現並優化了 **Xu et al. [31]** 的基礎工作，並將其應用於現代、大規模的多資產場景（台灣 50 指數成分股，2005-2025 年數據）。

### 1. 從序列架構轉向全 MLP 架構
我們將傳統的循環神經網絡 (LSTM) 轉向 **TSMixer** (Time-Series Mixer) —— 一種全 MLP (多層感知器) 架構。這使得模型能更有效地利用股票間的橫向相關性（Cross-sectional dependence），同時避免了 RNN 常見的計算負擔或梯度消失問題。

### 2. 混合參數 ($\lambda$) 的關鍵角色
我們研究的核心發現之一是：$\lambda$ 不僅僅是一個調優常數，更是一個 **戰略導航輪**。
- **高 $\lambda$**: 將模型拉向 GARCH 的穩定性，增強對尾部風險指標的表現。
- **低 $\lambda$**: 允許模型透過非線性表徵學習，優先提升平均預測準確率。
我們證明了透過精心選擇 $\lambda$，TSMixer 能穩定優於傳統的 GARCH 模型。

### 3. 多資產共享參數設計
透過 **參數共享 (Shared-Parameter)** 設計，我們的模型能彙整多個資產的資訊。這種設計能有效減弱特定股票的隨機噪聲，同時透過 GARCH-informed loss 保留資產本身的動態，使其在聯合波動度預測中表現卓越。

### 4. 對 Alpha 因子的高度警覺
我們的實證研究顯示，對於高維度的 **Alpha 因子 (Alpha Factors)** 應保持審慎。在沒有明確篩選或結構約束的情況下，盲目加入這些特徵產生的噪音往往會掩蓋波動度動態。這進一步強調了嵌入計量結構（如 GARCH）的重要性，而非單純依賴特徵工程的擴張。

---

## 📈 實驗背景
- **研究資產**：台灣 50 指數成分股。
- **時間維度**：2005 - 2025 年（涵蓋多個市場週期的長線評估）。
- **研究目標**：結合 GARCH 監督的多步波動度預測。

---

## 🏗️ 如何開始

有關環境配置、資料預處理流水線以及如何重現我們實驗結果的詳細步驟，請參閱：

👉 **[重現指南 (reproduction.md)](reproduction.md)**

---

## 👥 研究團隊

我們是來自 **國立台灣大學 (NTU)** 的研究小組，在Applied Deep Learning課程相遇並組成了小組，致力於探索金融 AI 的邊界：

- **蔡宇傑** (台大財金雙資料科學碩二, r13723046@ntu.edu.tw)
- **柯宥圻** (台大資料科學碩一, r14946005@ntu.edu.tw)
- **高大剛** (台大財金雙資工碩二, r12922156@ntu.edu.tw)
- **黃和謙** (台大資料科學碩一, r14946007@ntu.edu.tw)
- **Jovytta E. Yauwanta** (台大資工大三, b12902087@csie.ntu.edu.tw) 
- **郭宥真** (台大資工大三, b12902061@csie.ntu.edu.tw)

---

## 🔗 相關資源
- 📽️ **[介紹影片](https://www.youtube.com/watch?v=AhvbGsd1O98)**
- 📊 **[簡報檔案](https://docs.google.com/presentation/d/1RK-kOrt1dCIzXsmUMu1cpMOTFbq2gh3wWMnMSdsICV0/edit?usp=sharing)**
- 📁 **[訓練模型與實驗結果](https://drive.google.com/drive/u/0/folders/1xbHQe9j2vbTLrdpKLcS7qy-QH8qNLHXI)**
