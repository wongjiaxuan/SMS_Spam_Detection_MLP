# 📱 SMS Spam Detection System

A **Multi-Layer Perceptron (MLP) Neural Network**-based system for detecting SMS spam and fraudulent messages using **Backpropagation** and **TF-IDF** feature extraction.

---

## 🎯 Overview

This project implements an intelligent SMS spam detection system that can distinguish between legitimate (ham) and spam messages with **98.13% accuracy**. The system uses a neural network approach with:

- **Multi-Layer Perceptron (MLP)** with 4 hidden layers
- **Backpropagation** algorithm for training
- **TF-IDF Vectorization** for feature extraction
- **Web-based UI** for easy interaction

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🔍 **Real-time Prediction** | Analyze single SMS messages instantly |
| 📋 **Batch Testing** | Test multiple messages at once |
| 📊 **Performance Metrics** | View accuracy, precision, recall, F1-score |
| 📈 **Interactive Charts** | Class distribution, confusion matrix, feature importance |
| 🎨 **Modern Web UI** | Beautiful responsive interface |
| ⚡ **Fast Processing** | Cached charts for quick loading |

---

## 📊 Model Performance

| Metric | Score |
|--------|-------|
| **Accuracy** | 98.13% |
| **Precision** | 90.64% |
| **Recall** | 95.98% |
| **F1-Score** | 93.24% |

### Confusion Matrix

| Actual \ Predicted | Ham | Spam |
|--------------------|-----|------|
| **Ham** | 4751 | 74 |
| **Spam** | 30 | 717 |

---

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

### Step 1: Install Dependencies
```
pip install -r requirements.txt
```

### Step 2: Run the Web Application
```
python app.py
```

### Step 3: Open Browser
```
http://localhost:5001
```
