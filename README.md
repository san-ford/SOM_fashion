# Retrieval of Related Clothing Items Using a Self-Organizing Map
This app allows a user to upload an image to retrieve related clothing images. The images retrieved are from the MNIST fashion dataset. The retrieval algorithm uses a self-organizing map (an unsupervised machine learning algorithm) to categorize the clothing items based on the similarity of their visual features. To run the app, clone the repository and perform the following steps:

## Create Virtual Environment
Ensure you are in the "SOM_fashion" directory. To create a virtual environment for the app using conda, run the following commands
```bash
conda env create --file environment.yml
conda activate som_fashion_env
```

## Run App
```bash
python3 app.py
```
Once the app is running, use a browser to navigate to:

[http://127.0.0.1:5000/](http://127.0.0.1:5000/)

# The Self-Organizing Map Algorithm
This project is an exercise that explores the image association and visualization power of the Self-Organizing Map (SOM) algorithm. SOMs are useful for visualizing the relationships between data points in multidimensional space because they reduce dimensionality while preserving the topological structure. As an unsupervised learning algorithm, SOMs are well-suited for identifying clusters of similar data points without the need for labels.

In this project, I apply the scikit-learn package for SOMs to the MNIST fashion dataset, which contains images of clothing to reduce its dimensionality from 784 features to 2. This creates a map of the dataset where we see similar clothing items clustered together in the Jupyter Notebook. Although the labels in the dataset are not necessary to create the map, they are applied to show the clusters more explicitly.

# The Jupyter Notebook
The Jupyter Notebook shows an intuitive visualization of how the algorithm organizes the data. I have created a grid of images that correspond to the nodes of the self-organizing map. Each of these images are an average of all the images closest to the corresponding node in the map space. The visualization shows an intuitive grouping of similar images. For example, most of the footwear images were placed in the top-right corner of the map space by the algorithm.

![image](https://github.com/user-attachments/assets/2019a229-8cbd-4067-abc9-d61410b88a71)

Note: This image was created by the first run of the notebook. This is a continuing project.
