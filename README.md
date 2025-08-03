# Biomedical-Image-Technology-Project
A project developed in MATLAB to classify stages of Alzheimer's disease using brain MRI images. This work leverages the AlexNet Convolutional Neural Network (CNN) architecture to support the early and accurate diagnosis of the disease.

# About The Project
Alzheimer's disease is an irreversible, progressive brain disorder that slowly destroys memory and thinking skills. With the number of cases rising sharply in Indonesia, early detection is critical. While Magnetic Resonance Imaging (MRI) is a key tool for observing brain atrophy associated with Alzheimer's, manual analysis is time-consuming and requires expert radiologists.


This project addresses these challenges by developing an automated classification system using deep learning. The program is built in MATLAB R2024a and employs a Convolutional Neural Network (CNN) with the AlexNet architecture to classify brain MRI scans into four distinct stages of dementia. The goal is to provide a reliable, high-precision tool that can assist in the early diagnosis of Alzheimer's disease.

Classification Categories
The model is trained to differentiate between four stages of cognitive health:
1. Normal (Non-demented)
2. Very Mild Dementia 
3. Mild Dementia 
4. Moderate Dementia

# Key Features
1. Deep Learning Classification: Utilizes a CNN with the AlexNet architecture, fine-tuned for the specific task of Alzheimer's diagnosis.
2. MATLAB-Based Implementation: The entire workflow, from image processing to model training and evaluation, is developed within the MATLAB environment.
3. Comprehensive Preprocessing: Implements a multi-step image enhancement pipeline, including grayscale conversion, median filtering, CLAHE for contrast enhancement, and Gaussian smoothing.
4. Feature Extraction: Employs Histogram of Oriented Gradients (HOG) to capture the shape and structure of objects within the MRI images.
5. Transfer Learning: Leverages a pre-trained AlexNet model and adapts its final layers for the 4-class classification problem, accelerating training and improving performance.

# Dataset
1. Name: Open Access Series of Imaging Studies (OASIS)
2. Source: Washington University Alzheimer's Disease Research Center (ADRC)
3. Description: The project uses a cross-sectional T1-weighted MRI dataset from OASIS. The cohort includes 198 individuals aged 60 to 96, of whom 100 were diagnosed with very mild to moderate Alzheimer's disease.
4. Ground Truth Labeling: The classification labels are based on the Clinical Dementia Rating (CDR) score:
   a. CDR 0: Non-Demented
   b. CDR 0.5: Very Mild Demented
   c. CDR 1: Mild Demented
   d. CDR 2: Moderate Demented

# Methodology & Workflow
The project follows a systematic pipeline for model development:
1. Dataset Preparation: The OASIS MRI dataset is loaded. To handle class imbalance and prevent computational overload, the data for each of the four classes is balanced, with each class limited to 600 images.
2. Image Preprocessing: Each image undergoes a series of enhancement steps:
   a. Conversion to grayscale to reduce complexity.
   b. Median filtering to remove salt-and-pepper noise.
   c. Contrast Limited Adaptive Histogram Equalization (CLAHE) to improve local contrast.
   d. Gaussian smoothing to reduce high-frequency noise.
   e. Histogram of Oriented Gradients (HOG) feature extraction to capture structural information.
3. Data Splitting: The processed dataset is randomly divided into 80% for training and 20% for validation.
4. Model Configuration (Transfer Learning):
   a. A pre-trained AlexNet model is loaded.
   b. The final classification layers of AlexNet are replaced with new layers configured for the project's four output classes.

5. Data Augmentation: To prevent overfitting and improve model generalization, the training data is augmented with random rotations and translations.

6. Model Training: The fine-tuned AlexNet model is trained for 10 epochs using the SGDM (Stochastic Gradient Descent with Momentum) optimizer, a mini-batch size of 32, and an initial learning rate of 1e-4.


# Result

![WhatsApp Image 2024-05-27 at 11 35 41 (1)](https://github.com/user-attachments/assets/9c5114a3-71d9-4973-a756-3871b80fd083)

