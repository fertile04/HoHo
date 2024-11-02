Project 1: Battery State Estimation Using Transformer Architecture
Overview
Project Duration: March 2023 - July 2023
Objective: Develop a Transformer-based model for predicting voltage values to estimate the State of Charge (SOC) of a battery.
Location: Check the capstone folder in the repository for project files and code.
Tech Stack
Language: Python
Libraries: PyTorch, TensorFlow
Project Description
This project was part of a capstone design course aimed at predicting battery SOC using a Transformer deep learning model focused on voltage prediction. The Transformer model was compared against LSTM and GRU models, demonstrating superior accuracy in voltage estimation.

Results
Prediction Performance: The Transformer model outperformed LSTM and GRU models in prediction accuracy.



Project 2: Battery State Estimation Using Asymmetric-Transformer
Overview
Project Duration: September 2023 - September 2024
Objective: Enhance SOC estimation accuracy by developing an Asymmetric-Transformer model that leverages current derivative data.
Location: Check the SOC_ATNN_1dCNN folder in the repository for project files and code.
Tech Stack
Language: Python
Library: PyTorch
Project Description
This project focused on improving battery SOC estimation accuracy by analyzing the correlation in battery data and addressing the low correlation in SOC sequences. An Asymmetric-Transformer model was designed to utilize the rate of change in current data, reducing the number of model parameters by 47% and increasing estimation accuracy. Experimental results showed that this model achieved 110% better performance than LSTM and 93% better performance than the standard Transformer model.

Results
Performance Improvement: Achieved 110% improvement over LSTM and 93% over the standard Transformer in estimation accuracy.



Project 3: Battery State Estimation Using CNN-Transformer
Overview
Project Duration: January 2024 - Ongoing
Objective: Improve SOC estimation accuracy by leveraging the complementary strengths of CNN and Transformer layers.
Location: Check the SOC_ATNN_1dCNN folder in the repository for project files and code.
Tech Stack
Language: Python
Library: PyTorch
Project Description
This project combines CNN and Transformer layers to address both global and local data patterns for SOC estimation. The CNN layer captures spatial features and enhances input sequences with positional information, overcoming limitations of traditional Transformer models. The experiment showed that this model achieved 122% lower loss compared to the LSTM method.

Results
Loss Reduction: Achieved 122% lower loss compared to the LSTM model.
