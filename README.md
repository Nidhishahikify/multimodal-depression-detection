# Multimodal Depression Detection System

A **multimodal AI-based depression detection system** that analyzes both **facial expressions and acoustic/speech features** to identify patterns associated with depression. The system combines visual and audio information using machine learning techniques and provides an interactive web interface for analysis.

## 📌 Project Overview

Depression can manifest through multiple behavioral indicators, including changes in **facial expressions, speech patterns, and vocal characteristics**. This project aims to explore a multimodal approach by combining **facial expression analysis** with **acoustic feature extraction** to support automated depression detection.

The application processes input data, extracts relevant features from different modalities, applies trained machine learning models, and presents the prediction through a user-friendly web interface.

## ✨ Key Features

- 🎭 **Facial Expression Analysis** – Analyzes facial features and expressions from images/video input.
- 🎙️ **Acoustic Feature Extraction** – Extracts relevant features from speech/audio signals.
- 🤖 **Machine Learning-Based Prediction** – Uses trained ML models for depression-related classification.
- 🔀 **Multimodal Analysis** – Combines information from visual and acoustic modalities.
- 🌐 **Interactive Web Interface** – Provides an easy-to-use interface for submitting inputs and viewing results.
- ⚠️ **Error Handling** – Displays meaningful messages when invalid or unsupported input is provided.
- 🚀 **Deployment Support** – Configured for deployment using Render.

## 🛠️ Technologies Used

- **Programming Language:** Python
- **Web:** HTML, CSS
- **Machine Learning:** Scikit-learn
- **Audio Processing:** Acoustic feature extraction
- **Computer Vision:** Facial expression analysis
- **Backend:** Python
- **Frontend:** HTML/CSS

## 📂 Project Structure

```text
multimodal-depression-detection/
│
├── models/              # Trained machine learning models
├── scripts/             # Data processing and feature extraction scripts
├── templates/           # HTML templates for the web interface
├── tests/               # Application testing files
│
├── app.py               # Web application
├── main.py              # Main application logic
├── config.py             # Configuration settings
├── requirements.txt      # Python dependencies
├── runtime.txt           # Runtime configuration
├── DEPLOYMENT.md         # Deployment instructions
└── .gitignore            # Ignored files
