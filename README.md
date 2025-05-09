# Self-Organizing Map of Clothing Item Images
This is an exercise that explores the image association and visualization power of the Self-Organizing Map (SOM) algorithm. SOMs are useful for visualizing the relationships between data points in multidimensional space because they reduce dimensionality while preserving the topological structure. As an unsupervised learning algorithm, SOMs are well-suited for identifying clusters of similar data points without the need for labels.

Here, I apply the scikit-learn package for SOMs to the MNIST fashion dataset, which contains images of clothing to reduce its dimensionality from 784 features to 2. This creates a map of the dataset where we see similar clothing items clustered together. Although the labels in the dataset are not necessary to create the map, they are applied to show the clusters more explicitly.

For an intuitive visualization of how the algorithm organizes the data, I have created a grid of images that correspond to the nodes of the self-organizing map. Each of these images are an average of all the images closest to the corresponding node in the map space. This visualization shows an intuitive grouping of similar images. For example, most of the footwear images were placed in the top-right corner of the map space by the algorithm.
![image](https://github.com/user-attachments/assets/2019a229-8cbd-4067-abc9-d61410b88a71)

Note: This image was created by the first run of the notebook. This is a continuing project.
