# Diabetes_Prediction_ML-
This project predicts whether a person has diabetes using Machine Learning. It uses the PIMA Diabetes dataset, data preprocessing, and an SVM classifier for accurate results. Users can input medical values to get a real-time prediction. Built in Python for learning and practical use.
#  Diabetes Prediction System using Machine Learning

This project is a Machine Learning-based system that predicts whether a person is diabetic or non-diabetic using medical diagnostic data. Built using **Python**, **Support Vector Machine (SVM)**, and the **PIMA Diabetes Dataset**, this project demonstrates a complete ML workflow from preprocessing to prediction.

---

##  Features

* Predicts diabetes using medical parameters
* Uses Support Vector Machine (Linear SVM)
* Utilizes StandardScaler for data normalization
* Trained on PIMA Diabetes Dataset
* User-friendly script for input-based prediction
* Clean and modular code structure

---

##  Dataset

**PIMA Diabetes Dataset**

* Contains medical features such as:

  * Pregnancies
  * Glucose
  * Blood Pressure
  * Skin Thickness
  * Insulin
  * BMI
  * Diabetes Pedigree Function
  * Age
* Target variable: **0 – Non-Diabetic**, **1 – Diabetic**

---

##  Model Used

* **Support Vector Machine (SVM)** with a linear kernel
* Chosen because it performs well on binary classification tasks
* Trained after scaling data using StandardScaler

---

##  Technologies Used

* Python
* NumPy
* Pandas
* Scikit-learn

---

##  Example Prediction Code

```python
input_data = (7,196,90,0,0,39.8,0.451,41)
input_data_as_numpy_array = np.asarray(input_data)
input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
std_data = scaler.transform(input_data_reshaped)
prediction = classifier.predict(std_data)
```

---

##  Results

* Achieves strong accuracy on test data
* Generalizes well for unseen inputs

---

##  Project Highlights

* Demonstrates complete ML pipeline
* Good for learning classification models
* Easy to modify and extend

---

##  Contact

For any queries, feel free to reach out!
