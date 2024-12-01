# Employee Security System

## Overview

This project implements a **contactless employee check-in system** for Stark Industries, leveraging **smartphone sensor data** and **machine learning** for secure and seamless access control. The system replaces physical keycards with a smartphone-based gait analysis approach, ensuring enhanced security and convenience.

### Problem Statement

Currently, employees use physical keycards to access the building. However, this system is susceptible to:

- Loss or theft of keycards.
- Unauthorized access through stolen or cloned keycards.

To address these issues, this project proposes a machine-learning-based solution using **gait analysis**, which is unique to each individual, to authenticate employees.

---

## Features

- **Contactless Authentication**: Utilizes accelerometer data from employees' smartphones.
- **Gait Analysis**: Compares real-time gait patterns with historical patterns stored on the server.
- **Automated Access Control**: If the gait pattern matches, the system automatically opens the door for the employee.

---

## System Workflow

1. **Data Capture**: Employee's smartphone collects accelerometer data when entering the firm's premises.
2. **Data Transmission**: The smartphone sends the sensor data to the server.
3. **Pattern Analysis**: 
   - The server runs machine learning algorithms to analyze the received gait pattern.
   - Compares the pattern with the employee's historical gait data stored in the system.
4. **Decision**: 
   - If there is a match, access is granted, and the doors open.
   - If not, the employee is denied access.
5. **Feedback**: The system logs all successful and unsuccessful access attempts for auditing purposes.

---

## Technologies Used

- **Smartphone Sensors**: Accelerometer for gait data collection.
- **Machine Learning Algorithms**: For gait pattern analysis and authentication.
- **Server-Side Programming**: For data processing and decision-making.
- **Database**: To store employees' historical gait patterns securely.

---

## Prerequisites

- **Hardware**: Smartphones with accelerometers.
- **Software**:
  - Python (for ML models)
  - Flask/Django (for server API)
  - SQLite/PostgreSQL (for database storage)
- **Development Environment**:
  - Android/iOS for smartphone integration.
  - Machine Learning libraries: scikit-learn, TensorFlow/PyTorch.

---

## Installation

1. Clone the repository:  
   ```bash
   git clone https://github.com/username/employee-security-system.git
   cd employee-security-system


# Human Activity Recognition (HAR) Using Smartphones  

## Overview
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
