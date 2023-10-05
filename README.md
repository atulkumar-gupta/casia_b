# person-reid
person re-identification using deep learning 

## Pre-requisites
Listed below are some dependencies other than the ususal ones such as numpy, pandas, matplotlib etc.
+ PyTorch >= 0.3
+ Python >= 3.6

|Dataset|Folder|
|---|---|
|Casia-B|casia_b|
|Market 1501|st-reid| 
# casia_b

Casia-b dataset https://www.paperswithcode.com/dataset/casia-b </br>
nm-01 to nm-04 are used for training whereas nm-05 and nm-06 are used for validation and testing respectively.

Run the files in the following order as described in the table
|file|description|
|---|---|
|python3 preprocessing(1).ipynb|crop and center images in casia-b </br> by finding center of mass of person|
|dataset_creation_gei(2).ipynb|find all gait cycles in given set of images </br> hence find gait energy image|
|fine_tune_gei(3).ipynb|train models to indentify person id based on gait energy image|
|fine_tune_lat(4).ipynb|train model to indentify person based on latent space found|
|recons(5).ipynb|predict missing frames in gait cycle|

Helper files
|file|description|
|---|---|
|loess.py|Smoothening function for finding gait cycle images|
|pca.ipynb|project latent space into lower dimensional space for faster training and inference|
|strip_output.py|.ipynb file to .py file|
