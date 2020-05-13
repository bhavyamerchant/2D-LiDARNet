# 2D-LiDARNet
Originally created for Math 4339: Introduction to Data Science and Machine Learning at University of Houston with the Dr. Skripnikov and Dr. Poliak as instructors. Our purpose was to create a neural network model that could correctly identify vegetation vs no vegetation from cross sectional images from LiDAR. We obtained LiDAR scans of Enchanted Rock during the EAS Geophysics Summer Field Camp in 2018 and performed preprocessing methods. We tested a CNN and an ANN with two different sets of subsections - 48 by 48 pixels vs 28 by 28 pixels - to compare the performance between the two neural nets.
## Preprocessing and Organizing Data
### Preprocessing
The "Preprocessing.R" script shows how the preprocessing was performed. We first converted the LiDAR point cloud dataset into cross sectional images of a rock face using cloud compare. Then, we split up the cross-sections from 480 by 480 pixels into a hundred 48 by 48 pixel sub-sections. Then, we used a "black counter" for-loop to remove noise from the dataset so that the cross sections with more than 20 black pixels were used. In total, we ended up with 860 total cross sectional images.
### Organizing Data
Next, we manually sorted the training data and the test data with a 80/20 split. Then we used the resize.sh script to transform the subsections into 28 by 28 pixels. Finally, using the convert.py script, we transformed the two datasets into MNIST-like dataframes so that our models in R could easily read in the data.

##
