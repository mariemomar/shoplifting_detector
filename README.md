# 🛒 Shoplifting Detection System

A deep learning-based system for detecting shoplifting behavior in surveillance videos using Django and PyTorch.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Django](https://img.shields.io/badge/Django-4.2-green)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0-orange)


## 🎯 Overview

This project provides a REST API for detecting shoplifting activities in surveillance videos using a hybrid EfficientNet-B0 + LSTM neural network architecture. The system processes video frames to identify potential shoplifting behavior with confidence scoring.

## ✨ Features

- **🎥 Video Analysis**: Process MP4, AVI, MOV, and MKV files
- **🤖 Deep Learning**: EfficientNet-B0 + LSTM model for temporal analysis
- **🚀 REST API**: Fully documented Django REST Framework endpoints
- **📊 Confidence Scoring**: Probability-based detection results
- **🎨 Web Interface**: Modern, responsive frontend for easy testing
- **⚡ Real-time Processing**: Fast inference with GPU acceleration support
- **🔒 Security**: File validation and error handling

## 🧠 Model Architecture
EfficientNet-B0 (Pre-trained) → Feature Extraction
↓
LSTM (128 units) → Temporal Analysis
↓
Fully Connected → Binary Classification
↓
Sigmoid → Shoplifting Probability

**Key Components:**
- **Backbone**: EfficientNet-B0 (ImageNet pre-trained)
- **Sequence Modeling**: LSTM with 128 hidden units
- **Input**: 16 frames per video (224x224 resolution)
- **Output**: Binary classification (shoplifting vs non-shoplifting)

## 🚀 Installation

### Prerequisites

- Python 3.8+
- PyTorch 2.0+
- Django 4.2+

### Step-by-Step Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd shoplifting_detector

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
3. **Setup environment variables**
   ```bash
   # Create .env file
   echo "SECRET_KEY=your-secret-key-here" > .env
   echo "DEBUG=True" >> .env
   echo "ALLOWED_HOSTS=localhost,127.0.0.1,0.0.0.0" >> .env
4. **Run migrations**
   ```bash
   python manage.py migrate
## ✔ How to run
   ``` bash  
  python manage.py runserver 
