# Farmer Chatbot Project

This README provides step-by-step instructions to run the Farmer Chatbot project in **Google Colab**. You can either run it directly in cells or execute the main logic using `farmer_chatbot.py`.

## 📌 Prerequisites:

1. Google Colab account
2. Python 3.x
3. Required libraries:

   * `torch`
   * `transformers`
   * `Pillow`
   * `numpy`
   * `fastapi`
   * * `gradio`

You can install missing libraries in Colab using:

```python
!pip install torch transformers pillow numpy fastapi
```

---

## 📂 Folder Structure:

```
project-folder/
│
├── farmer_chatbot.py          # Main chatbot script
├── crop_health_model.pth      # Trained model for crop health
├── processor.pkl              # Image processor
├── sample_images/             # Folder containing sample images
│   └── test_image.jpg
└── README.md                  # This documentation
```

---

## 🚀 Running the Project in Google Colab:

### 🔹 Step 1: Mount Google Drive

First, mount your Google Drive to access your project files:

```python
from google.colab import drive
drive.mount('/content/drive')
```

---

### 🔹 Step 2: Navigate to Project Directory

Make sure to navigate to your project folder:

```python
%cd /content/drive/MyDrive/YourProjectFolder
```

Replace `YourProjectFolder` with the actual folder name.

---

### 🔹 Step 3: Import Dependencies

```python
import torch
from transformers import AutoModelForImageClassification, AutoProcessor
from PIL import Image
import numpy as np
from farmer_chatbot import classify_crop_health, get_crop_health_advice
```

---

### 🔹 Step 4: Load the Model and Processor

```python
image_model = AutoModelForImageClassification.from_pretrained('path_to_your_model')
processor = torch.load('processor.pkl')
```

Replace `path_to_your_model` with the correct path to your model.

---

### 🔹 Step 5: Test with an Image

```python
image_path = '/content/drive/MyDrive/YourProjectFolder/sample_images/test_image.jpg'
image = Image.open(image_path)

disease = classify_crop_health(np.array(image))
print(f'Detected Disease: {disease}')

advice = get_crop_health_advice(np.array(image))
print(f'Chatbot Advice: {advice}')
```

---

## 🏃 Running with `farmer_chatbot.py`(GRADIO UI )

If you want to run everything directly from `farmer_chatbot.py`, use:

```bash
!python farmer_chatbot.py
```

This will execute the main logic directly, assuming it has the right calls to `classify_crop_health` and `get_crop_health_advice`

---
