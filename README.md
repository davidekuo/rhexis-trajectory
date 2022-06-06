# 👁⚕️ Capsulorrhexis Trajectories for Automated Surgical Training Feedback 

![GitHub](https://img.shields.io/badge/CS231n-Final%20Project-red) ![GitHub](https://img.shields.io/badge/CS229-Final%20Project-red)

## 💻 Contributors 
David Kuo MD, Ben Viggiano, Riya Sinha, Ben Ehlert

## ⚡️ Gettting Started 
1) Clone the repository to your Google Drive.<br>  

2) Download our dataset folder from this [Google Drive Link](https://drive.google.com/drive/folders/1QUk7AXNivhF9SRqwJA2lCihnp-nO8Juh?usp=sharing) and place the folder in another location within your Google Drive

3) Open the `.ipynb` file you would like to utilize in Google CoLab and follow the instructions.

![Alt Text](https://github.com/davidekuo/rhexis-trajectory/blob/main/rhexis_trajectory_output.gif)

## Summary
The quality and volume of surgical training can vary widely between different training institutions, and can be significantly affected by unexpected circumstances such as the recent (and still ongoing) COVID-19 pandemic that saw operating rooms across the world close except for emergent cases. As such, there is a great need for trainees to maximize the educational value of every surgical case they perform. In recent years, internet platforms such as YouTube have enabled surgeons to share cases and discuss new techniques, and allowed trainees to receive valuable feedback from more experienced surgeons on their own cases; however, these initiatives are limited by the availability of experienced surgeons with the time and willingness to review trainee-submitted cases outside of working hours and provide feedback. To address this limitation, we propose a machine learning platform that provides automated feedback for trainee-submitted surgical videos, specifically cataract surgery. 

For the purposes of this project, we focus solely on the capsulorrhexis step of cataract surgery: in this step, the surgeon uses a forceps to peel a 5.5-6mm circle in the 15um thick anterior lens capsule (akin to peeling a circle in the skin of a grape) through a 2mm main incision in a space that is roughly 4mm deep and 7mm x 7mm horizontally. The capsulorrhexis is particularly challenging for training cataract surgeons because it is one of two critical steps in which complications most commonly occur, and uniquely stresses the foundational skills of 'floating' and 'pivoting' within the 2mm main incision necessary for maneuvering safely and effectively within the eye.


Our project is composed of two parts:
* **For CS231n:** 
  * Keypoint detection using a Detectron 2 Model
  * Semantic Segmentation utilizing the [MICCAI2021 Cataract Semantic Segmentation Model](https://github.com/RViMLab/MICCAI2021_Cataract_semantic_segmentation) from [Robotics and Vision in Medicine Lab at King's College of London](https://rvim.online/)

* **For CS229:** 
  * Feature engineering and feature selection with PCA from the output of our 231n model
  * Trajectory Classification for feedback using various methods (Logistic Regression, Quadratic GDA, and MLP Classifier)



An illistration of our full project is shown below:

![alt text](https://github.com/davidekuo/rhexis-trajectory/blob/main/ModelOverview.png)



## CS231n Project
The focus of our CS231N project is to develop a keypoint detection model that tracks the tips of the utrada forceps instrument over the course of the capsulorrhexis in order to generate instrument trajectories. These instrument trajectories will then be fed to downstream models (the focus of our CS229 project) to generate quantitative and automated feedback for training cataract surgeons. Additionally, we employ a pretrained semantic segmentation model to generate additional features of interest such as pupil center and boundaries as well as incision location (which can be more challenging to label with keypoints) to further enrich our feedback.

The inputs to our keypoint detection model are JPEG images generated from the frames of MP4 videos of the capsulorrhexis step of cataract surgery, along with ground-truth bounding box and keypoint labels in the COCO JSON format. We then use an R-CNN-FPN model to output bounding box and keypoint predictions for the utrada forceps and utrada forceps tips respectively for each frame of the input video. We then concatenate and plot the predicted keypoints to generate instrument trajectories for each surgical video.
