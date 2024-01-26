#  Priviliged Knowledge Distillation with Optimal Transport (PKDOT)



----------------
----------------

# 0. Table of Contents
#### This manual has has two parts
### 1- Installation
Provides installation guide.
### 2- Running the code
Details on how to run the code.

----------------
# 1. Installation
##### 1.2 Pytorch(1.10.0) which can be installed with details provided here: https://pytorch.org/get-started/locally/
For most users, ```pip3 install torch torchvision``` should work.
If you are using Anaconda, ```conda install pytorch torchvision -c pytorch``` should work. 

Create a virtual environment using Conda or Virtualenv and install all the dependencies


# 2. Running The code
### 2.1 Important Files In codebase: 
#### 2.1.1 `models.py` in the 'models' folder creates and defines all the models.
#### 2.1.2 `pkdot_kfold.py` The main code. Trains the student model.	
#### 2.1.3 `video_dataset_mm.py` Provides the dataloaders to be used by pkdot_kfold file. Used to load both visual and phyioslogical modality.
#### 2.1.4 `pkdot_utils.py` Provides functions for similarity matrices and visualizations.




## 2.3 Running
The first step is to create the folder heirarchy.


 
1) Run the 'face_detector.py' file to extract facial frames from raw videos and save them in the 'Cropped _Aligned' folder.
2) Run the 'audio_extract.py' file to extract audio segments from raw audios and save them into separate folders.







### Reference

```
