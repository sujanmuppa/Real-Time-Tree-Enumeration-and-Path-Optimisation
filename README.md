# Real-Time-Tree-Enumeration-and-Path-Optimisation

# Land Cover Segmentation and Pathfinding System

## 📌 Overview
This project consists of **two main components**:

1. **Model Training**: Train a **SegFormer-based segmentation model** for land cover classification.
2. **Flask Local Deployment**: Deploy the trained model using a Flask-based web application with **pathfinding algorithms (A* and Dijkstra)**.

---

## 📂 Directory Structure
```
Land-Cover-Segmentation/
├── Model-Training/
│   ├── Training.py
│   ├── requirements.txt
│   ├── GoodTrain.pth  # (Generated after training)
│
├── Flask-Local-Implementation/
│   ├── model/
│   │   ├── GoodTrain.pth  # (Copy from Model-Training)
│   ├── templates/
│   │   ├── landing.html
│   │   ├── main.html
│   ├── app.py
│   ├── requirements.txt
```

---

# 🚀 1. Model Training

## 📌 Step 1: Download Dataset
The dataset for training is available on **Kaggle**:
👉 [DeepGlobe Land Cover Classification Dataset](https://www.kaggle.com/datasets/balraj98/deepglobe-land-cover-classification-dataset)

1. Download the dataset from Kaggle.
2. Extract the dataset and **place all files inside `Model-Training/`**.

## 📌 Step 2: Install Dependencies
Navigate to the `Model-Training/` directory and install required libraries:
```bash
cd Model-Training
pip install -r requirements.txt
```

## 📌 Step 3: Train the Model
Run the training script:
```bash
python Training.py
```
- The model will train using **SegFormer**.
- After training, **GoodTrain.pth** will be saved in `Model-Training/`.

## 📌 Step 4: Move the Model for Deployment
Once training is complete, move the trained model to the Flask directory:
```bash
mv GoodTrain.pth ../Flask-Local-Implementation/model/
```

---

# 🌍 2. Flask Local Deployment

## 📌 Step 1: Install Dependencies
Navigate to the **Flask-Local-Implementation** directory and install required libraries:
```bash
cd ../Flask-Local-Implementation
pip install -r requirements.txt
```

## 📌 Step 2: Start the Flask Server
Run the Flask application:
```bash
python app.py
```

If successful, the terminal will show an output like:
```
 * Running on http://127.0.0.1:5000/
```

## 📌 Step 3: Open in Browser
- Open your browser and visit **[http://127.0.0.1:5000/](http://127.0.0.1:5000/)**.
- You should see the **Segmentation and Pathfinding interface**.

## 📌 Features Available:
✅ Upload a satellite image for **land cover segmentation**.  
✅ Compute **optimal paths** between selected points using **A* and Dijkstra**.  
✅ View **land cover classification** percentages.  
✅ Detect **number of trees** in the image.  

---

# 🔥 Troubleshooting

### 1️⃣ **Missing Dependencies**
If you face import errors, ensure all dependencies are installed:
```bash
pip install -r requirements.txt
```

### 2️⃣ **Model Not Found**
If you see `FileNotFoundError: GoodTrain.pth not found`:
- Ensure you have **copied GoodTrain.pth** into `Flask-Local-Implementation/model/`.
- If missing, retrain the model in `Model-Training/`.

### 3️⃣ **Port Already in Use**
If Flask cannot start due to **port 5000 being occupied**:
```bash
python app.py --port 5001
```

---

# 🎯 Conclusion
By following these steps, you can **train a deep learning model** for satellite image segmentation and **deploy it locally using Flask** for real-time analysis and pathfinding!

---

💡 **Next Steps**:
- Try deploying the model on a **cloud platform** (AWS, GCP, or Heroku).
- Fine-tune the segmentation model with **custom datasets**.

🚀 **Happy Coding!**
