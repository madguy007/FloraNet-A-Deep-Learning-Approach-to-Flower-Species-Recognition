# 🌸 FloraNet: Flower Classification using Deep Learning

## 📌 Overview

FloraNet is a deep learning-based image classifier trained to identify **102 flower species** using PyTorch and transfer learning.

Developed as part of Udacity AI Nanodegree.

---

## 🧠 Model Details

- Pretrained Model: VGG16
- Custom classifier with:
  - Fully connected layers
  - ReLU activation
  - Dropout regularization
- Test Accuracy: **83.5%**

---

## 📂 Project Structure

```bash
notebook/
  └── flower_classifier.ipynb

src/
  ├── predict.py
  └── utils.py

model/
  └── checkpoint.pth

assets/
  ├── sample_input.jpg
  └── sample_output.png

``` 
pip install -r requirements.txt
```
python src/predict.py --image_path assets/sample_input.jpg
```
📸 Sample Output
The model predicts top flower classes with probabilities.

🧰 Tech Stack
Python
PyTorch
Torchvision
NumPy
PIL

🔮 Future Improvements
Convert into web app (Streamlit)
Improve accuracy using ResNet

👨‍💻 Author

Madan Dahiphale
LinkedIn: www.linkedin.com/in/madandahiphale
