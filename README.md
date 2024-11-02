# Project 1: Battery State Estimation Using Transformer Architecture
### 📅 Project Duration
March 2023 - July 2023
### 🎯 Objective
Develop a Transformer-based model to predict voltage values for estimating battery SOC.
### 🛠 Tech Stack
Language: Python
Libraries: PyTorch, TensorFlow
### 📂 Project Location
Files are located in the capstone folder of this repository.
### 📜 Project Description
This capstone project uses a Transformer architecture to focus on voltage prediction for SOC estimation. The model’s predictions were compared with those from LSTM and GRU models, showing superior accuracy in voltage estimation through the Transformer structure.
### 📈 Results
Prediction Accuracy: The Transformer model outperformed both LSTM and GRU models in SOC prediction accuracy.






-----------------

# Project 2: Battery State Estimation Using Asymmetric-Transformer
### 📅 Project Duration
September 2023 - September 2024
### 🎯 Objective
Enhance SOC estimation accuracy with an Asymmetric-Transformer model that leverages current derivative data.
### 🛠 Tech Stack
Language: Python
Library: PyTorch
### 📂 Project Location
Files are located in the SOC_ATNN_1dCNN folder of this repository.
### 📜 Project Description
This project aimed to improve SOC estimation accuracy by developing an Asymmetric-Transformer model. By focusing on the rate of change in current data, the model reduced parameters by 32% while achieving higher accuracy. Compared to traditional LSTM and standard Transformer models, this approach provided substantial performance gains.
### 📈 Results
Performance Improvement: Achieved 52% improvement over LSTM and 48% over standard Transformer in accuracy.

-----------------

# Project 3: Battery State Estimation Using CNN-Transformer
### 📅 Project Duration
January 2024 - Ongoing
### 🎯 Objective
Combine CNN and Transformer strengths to improve SOC estimation accuracy by capturing both global and local data features.
### 🛠 Tech Stack
Language: Python
Library: PyTorch
### 📂 Project Location
Files are located in the SOC_ATNN_1dCNN folder of this repository.
### 📜 Project Description
This project leverages CNN and Transformer layers to simultaneously address global and local patterns in SOC estimation. The CNN layers enhance positional information in the input data, effectively overcoming Transformer limitations. Experimentally, this model achieved a 44% reduction in loss compared to the LSTM model.
### 📈 Results
Loss Reduction: 44% lower loss than the LSTM model, indicating improved SOC prediction accuracy.
