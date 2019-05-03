This is the final project for CS_5350 written by Janaan Lake April 2019.
This project contains several methods for creating a movie recommender system.
The dataset used for this project is the MovieLens dataset found at 
    http://files.grouplens.org/datasets/movielens/
The 100K dataset was used.  However, this project also has capability to run the 1M size dataset.

The different methods used for the recommender system are:
    * Average rating
    * Content-based (Based on the movie genres)
    * Collaboritive Filtering:  user-item and item-item
    * Matrix Factorization based on the method proposed by Simon Funk in the Netflix prize
    * Hybrid method
    * Neural Network

The accuracy and the MSE of the ratings predictions for each of these methods is printed for each test run.  Also,
the accuracy of the ratings are further broken down by each rating (1-5).

This project also has the capability to reduce the size of the dataset and increase the density of the utility 
matrix.  The utility matrix is an m x n matrix with m users and n items, with each element representing the rating
that user m gives to movie n.  It is created based on the information in the training set, which is the ratings
file provided in the download.  This matrix is a sparse matrix, which provides challenges for creating the 
recommender system.

Another test run is shown with the reduced utility matrix.
