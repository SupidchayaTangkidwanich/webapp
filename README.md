## Web Application for car part prediction and damage estimation with single image 

![pic](pj.jpg)  


### Step 1 : Create Environment 
### Install miniconda [link](https://docs.conda.io/en/latest/miniconda.html#)
 - conda create --name (your env name) python=3.8
 - conda activate (your env name)
### Step 2 : Git Clone 
 - git clone https://github.com/Umaporn19/webapp.git
### Step 3 : Install Packages
 - cd webapp
 - pip install flask
 - pip install pillow
 - pip install tensorflow
 - pip install numpy
 - git clone https://github.com/Wanita-8943/efficientnet_keras_transfer_learning.git
 - cd efficientnet_keras_transfer_learning
 - pip install -e .
 - pip install scikit-image
 ### Step 4 : Open WebApp
 - cd webapp
 - python server.py
 
