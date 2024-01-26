#  Priviliged Knowledge Distillation with Optimal Transport (PKDOT)



----------------
----------------

# 0. Table of Contents
#### This manual has has two parts
### 1- Installation
Provides installation guide.
### 2- Dataset 
Details on structuring and preparing the data.

### 2- Running the code
Details on how to run the code.

----------------
# 1. Installation
##### 1.2 Pytorch(1.10.0) which can be installed with details provided here: https://pytorch.org/get-started/locally/
For most users, ```pip3 install torch torchvision``` should work.
If you are using Anaconda, ```conda install pytorch torchvision -c pytorch``` should work. 

Create a virtual environment using Conda or Virtualenv and install all the dependencies using requirement.txt file.

# 2. Dataset
The first step is to create the folder heirarchy.
The dataset is originally split by subject. You should create separate directories per class. 

Biovid

 * 0 # class folder (BL1)
   * 071309_w_21-BL1-082  # Subject folder
     * img_00001.jpg #face images
     * .
     * img_00075.jpg #face images
   * .
   * 110810_m_62-BL1-094
     * img_00001.jpg #face images
     * .
     * img_00075.jpg #face images*        
 * 4 # Class folder (PA4)
   * 071309_w_21-PA4-006  # subject folder
     * img_00001.jpg #face images
     * .
     * img_00075.jpg #face images
   * 071614_m_20-PA4-010
     * img_00001.jpg #face images
     * .
     * img_00075.jpg #face images
    
  
Create annotaions files using create_annotationstxt.py file.
The annotations should be in the following format.
class_folder/subject_folder start_frame end_frame class label
e.g. 4/073109_w_28-PA4-062 0 75 1
The create_annotationstxt.py file also provides the code for k-fold annotations.
 


# 3. Running The code
### 3.1 Important Files In codebase: 
#### 3.1.1 `models.py` in the 'models' folder creates and defines all the models.
#### 3.1.2 `pkdot_kfold.py` The main code. Trains the student model.	
#### 3.1.3 `video_dataset_mm.py` Provides the dataloaders to be used by pkdot_kfold file. Used to load both visual and phyioslogical modality.
#### 3.1.4 `pkdot_utils.py` Provides functions for similarity matrices and visualizations.
#### 3.1.5 `physio_transforms.py`Provides the functions for transformation and filtering of physiological modality.


The 'pkdot_kfold.py' file requires the paths for the pretrained teacher models. 

### Reference

```
