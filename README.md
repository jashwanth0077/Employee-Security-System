# Human Activity Recognition (HAR) Using Smartphones  

## Project Overview  
This project involves the implementation of a machine learning model to perform **Human Activity Recognition (HAR)** using data collected from the accelerometer and gyroscope sensors of smartphones. The dataset used is the **UCI HAR Dataset**, which contains raw inertial sensor signals as well as precomputed time-domain and frequency-domain features for six different physical activities:  

- Walking  
- Walking Upstairs  
- Walking Downstairs  
- Sitting  
- Standing  
- Laying  

The goal of this project is to classify these activities based on the sensor data, demonstrating the effectiveness of machine learning techniques in activity recognition tasks.

---

## Why This Project?  
Human Activity Recognition is crucial in various fields, such as:  
- **Healthcare**: Monitoring patients' physical activity levels.  
- **Fitness Tracking**: Providing real-time feedback on workouts and physical performance.  
- **Smart Homes**: Automating tasks based on user activity.  
- **Human-Computer Interaction**: Enhancing gesture-based control systems.
- **Employee-Security-System**: For automatic verification or any detection of suspicious activity

By classifying activities, this project highlights the integration of machine learning with wearable and mobile sensors in real-world applications.

---

## Key Steps in the Project  

### 1. Data Preprocessing  
- **Loading Data**: Raw inertial sensor signals (accelerometer and gyroscope data) are read from text files.  
- **Precomputed Features**: Precomputed features such as mean, standard deviation, and FFT components are loaded for use in classification.  
- **Normalization**: Feature values are normalized using `StandardScaler` to improve model performance.  

### 2. Model Training  
A **Random Forest Classifier** is trained using the preprocessed features:  
- **Training Data**: Used to fit the model.  
- **Testing Data**: Used to evaluate the model's accuracy and generalization ability.  

### 3. Evaluation  
The model's performance is evaluated through:  
- **Accuracy Score**: Measures overall performance.  
- **Classification Report**: Provides precision, recall, and F1-score for each activity.  
- **Confusion Matrix**: Visualizes misclassifications.  

### 4. Feature Importance  
The most significant features contributing to the classification task are identified and visualized.

---

## Results  
The model achieves high accuracy in predicting the six activities, demonstrating the effectiveness of the chosen features and model. This project highlights the potential of sensor data and machine learning in activity recognition tasks.  

---
