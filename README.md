
# 📘 Fake News Detection Using DistilBERT

### **Cross-Domain Fake News Classification with Advanced Evaluation Metrics**

**Project ID: 25SK16** | **Course: CS3103 — Machine Learning Project**

---

## ⭐ Overview

This project builds a **robust, cross-domain Fake News Detection system** using a fine-tuned **DistilBERT transformer model**.
It performs:

* ⚡ High-accuracy binary classification → *Fake (0)* or *Real (1)*
* 🌍 Cross-domain testing on unseen datasets
* 📊 Advanced ML metrics (Macro-F1, AUROC, ECE, Brier Score, Robustness Indices)
* 🧠 Explainability via LIME
* 🎨 Full Streamlit dashboard with 20+ analytics graphs

---

# 📁 Folder Structure

```
Fake-News-Detection/
│
├── data/
│   ├── fake.csv
│   ├── True.csv
│   ├── gossipcop_fake.csv
│   ├── gossipcop_real.csv
│   ├── train (3).csv
│   ├── valid.csv
│   └── combined_news.csv        # Final cleaned training dataset
│
├── models/
│   └── distilbert_generalized/  # Trained checkpoint (BERT tokenizer + model)
│
├── artifacts/
│   ├── training_history.csv
│   ├── classification_report.csv
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   └── calibration_bins.csv
│
├── artifacts_test/
│   ├── classification_report_test.csv
│   ├── confusion_matrix_test.png
│   ├── roc_curve_test.png
│   ├── accuracies.csv
│   └── combined_summary.json
│
├── app.py                       # Streamlit Visualization Dashboard
├── train_distilbert.py          # Main Training + Evaluation Script
├── prepare_dataset.py           # Dataset merging & cleaning
│
├── requirements.txt
└── README.md
```

---

# 📦 Datasets Used

The project uses **three industry-standard Fake News datasets**:

---

### **1️⃣ LIAR Dataset**

* Short political statements with verdict labels
* 🔗 **ACL Paper:** [https://aclanthology.org/P17-2067/](https://aclanthology.org/P17-2067/)
* 🔗 **Kaggle:** [https://www.kaggle.com/datasets/armagansalman/liar-dataset](https://www.kaggle.com/datasets/armagansalman/liar-dataset)

---

### **2️⃣ ISOT Fake News Dataset**

* Real & fake news from mainstream media
* 🔗 **Official Dataset:** [https://www.uvic.ca/engineering/ece/isot/datasets/fake-news/index.php](https://www.uvic.ca/engineering/ece/isot/datasets/fake-news/index.php)
* 🔗 **Kaggle:** [https://www.kaggle.com/datasets/emineyetm/fake-news-detection-datasets](https://www.kaggle.com/datasets/emineyetm/fake-news-detection-datasets)

---

### **3️⃣ GossipCop (FakeNewsNet)**

* Fake celebrity news dataset
* 🔗 **Research Paper:** [https://arxiv.org/abs/1809.01286](https://arxiv.org/abs/1809.01286)
* 🔗 **GitHub:** [https://github.com/KaiDMML/FakeNewsNet](https://github.com/KaiDMML/FakeNewsNet)

---

# 🧹 Dataset Preparation Pipeline (prepare_dataset.py)

The script:

* Automatically detects **text** & **label** columns
* Merges 6+ datasets
* Standardizes labels to **0 = Fake, 1 = Real**
* Cleans and saves the final file:

```
data/combined_news.csv
```

---

# 🔥 Methodology 

### **1️⃣ Raw Datasets Collection**

LIAR, ISOT, GossipCop datasets imported and merged.

### **2️⃣ Label Mapping & Column Detection**

Automatic detection of:

* Correct text column
* Correct label column

### **3️⃣ Cleaned Combined Dataset**

Final file with:

* ~67,000 cleaned records
* Balanced fake/real labels

### **4️⃣ BERT Tokenization**

Using:

* DistilBertTokenizerFast
* 192 max sequence length

### **5️⃣ DistilBERT Fine-Tuning**

Training on:

* 2 epochs
* AdamW optimizer
* Linear warmup scheduler
* Mixed precision (AMP)

### **6️⃣ Training Evaluation**

Metrics computed:

* Accuracy
* Macro-F1
* Balanced Accuracy
* AUROC
* Confusion Matrix
* ROC Curve

### **7️⃣ Cross-Domain Test Evaluation**

Generalization tested on **train(3).csv**:

* Reports all metrics again
* Computes domain-shift robustness:

  * Accuracy Drop
  * AUROC Drop
  * Worst-Group Accuracy (WGA)

### **8️⃣ Streamlit Visualization**

Interactive dashboard with:

* Heatmaps
* ROC curves
* Training history
* LIME explanation
* WordCloud
* Probability graphs

---

# 🧠 Model Architecture (DistilBERT)

* 6 transformer layers
* 66M parameters
* Pre-classifier dense layer (ReLU)
* Final classification head → 2 outputs

---

# 📊 Evaluation Metrics

### **Model Performance Metrics**

* Accuracy
* Precision
* Recall
* Macro-F1
* AUROC
* Balanced Accuracy

### **Visualizations Saved**

* Confusion Matrix
* ROC Curve
* PR Curve
* Loss vs Epochs
* Label Distribution
* Text-Length Distribution

---

# ▶️ How to Run the Project

---

## **1️⃣ Install Requirements**

```bash
pip install -r requirements.txt
```

---

## **2️⃣ Prepare Dataset**

```bash
python prepare_dataset.py
```

Output:

```
data/combined_news.csv
```

---

## **3️⃣ Train DistilBERT**

```bash
python train_distilbert.py \
  --train_file data/combined_news.csv \
  --test_file "data/train (3).csv" \
  --epochs 2 \
  --batch_size 32
```

---

## **4️⃣ Run Streamlit Dashboard**

```bash
streamlit run app.py
```

Open the URL shown in the terminal.

---


---





