
# coding: utf-8

# In[165]:


"""
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

"""


# In[166]:


import pandas as pd
import numpy as np
import math
import tensorflow as tf
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt


# In[167]:


def download_data(path):
    """
    Downloads the 100k movielens dataset and unzips the files.
    Inputs:
    - path: The path where the movielens dataset will be downloaded
    """
    import requests
    import zipfile
    import os

    # download file
    resp = requests.get('http://files.grouplens.org/datasets/movielens/ml-100k.zip', allow_redirects=True, stream=True)

    if resp.status_code == 200:
        print('Successfully downloaded the data')
    elif resp.status_code == 404:
        print('File Not Found.  Could not download the dataset.')
    
    filename = 'ml-100k.zip'
    zfile = open(filename, 'wb')
    zfile.write(resp.content)
    zfile.close()

    zipf = zipfile.ZipFile(filename, 'r')  
    zipf.extractall(path)
    zipf.close()

    os.remove(filename)


# In[168]:


def clean_data(items, ratings):
    """
    Helper function for the 1M size dataset.
    
    Inputs:
    - items:  Pandas dataframe containing the movie data
    - ratings: Pandas dataframe containing the rating information
    
    Returns:  The same dataframes that have been reprocessed.
    """
    
    #Cleans up the items and ratings data so movies that aren't rated are discarded and renumbered
    print("The unique number of movies in the ratings file is " + str(ratings['movie_id'].nunique()))
    print("The total number of movies in the items file is " + str(items.shape[0]))
    items_new = items.loc[items['movie_id'].isin(ratings['movie_id'])]
    print("The reduced number of items in the items file is now " + str(items_new.shape[0]))

    max_id = items['movie_id'].max()
    new_movie_id = np.arange(1, items_new.shape[0]+1)
    new_movie = pd.Series(new_movie_id)
    new_movie.rename('new_id')
    items_new = items_new.assign(new_id=new_movie.values)
    
    for i in range(max_id + 1):
        new_id = items_new.loc[items['movie_id'] == i]['new_id']
        if (len(new_id) != 0):
            new = new_id.values
            new_movie_ids = ratings['movie_id'].mask(ratings['movie_id']==i, new)
            ratings = ratings.assign(movie_id = new_movie_ids)

    return items_new, ratings


# In[169]:


def load_data(path, data_size='small'):
    """
    Loads the dataset into pandas dataframes.
    
    Inputs:
    - path:  The path where the datasets are located.
    - datasize: {'small', 'large'}.  The small dataset is the 100k dataset, and the large
      dataset is the 1m dataset.  They each are processed slightly different.
      
    Returns:
     - users, items, ratings:  These are all separate pandas dataframes for the users, movies and ratings.
    """

    u_cols = ['user_id', 'age', 'sex', 'occupation', 'zip_code']
    r_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
    
    #Read in 100K data files
    if data_size=="small":
    
        users = pd.read_csv('ml-100k/u.user', sep='|', names=u_cols, encoding='latin-1')
        ratings = pd.read_csv('ml-100k/u.data', sep='\t', names=r_cols, encoding='latin-1')
        i_cols = ['movie_id', 'title' ,'release date','video release date', 'IMDb URL', 'unknown', 'Action', 'Adventure',
              'Animation', 'Children\'s', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
              'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
        items = pd.read_csv('ml-100k/u.item', sep='|', names=i_cols, encoding='latin-1')
    
    #Read in the 1M data files
    else:
        users = pd.read_csv('ml-1m/users.dat', sep='::', names=u_cols, encoding='latin-1', engine='python')
        ratings = pd.read_csv('ml-1m/ratings.dat', sep='::', names=r_cols, encoding='latin-1', engine='python')
        i_cols = ['movie_id', 'title', 'genre']
        items = pd.read_csv('ml-1m/movies.dat', sep='::', names=i_cols, encoding='latin-1', engine='python')
        
        items, ratings = clean_data(items, ratings)
        
    
    return users, items, ratings


# In[260]:


def process_data_for_NN(ratings, users, items, threshold=100):
    """
    Processes the data for use in the neural network.
    Genres are added to each movie in the ratings dataframe.
    User information, such as age and sex are added to the ratings dataframe.
    Removes the user_id and movie_id from each rating sample.
    Each input into the neural network will contain this information.
    
    Inputs:
    - ratings:  Pandas dataframe containing the ratings data
    - items:  Pandas dataframe containing the movie data
    - threshold:  The value used for reducing the items and users in the dataset.
    
    Returns:
    - x_train, y_train, x_test, y_test:  The processed data split into the training and testing set that
      can be used in the neural network.
    
    """
    
    ratings_merge = pd.merge(ratings, users, on='user_id', how = 'left')
    ratings_merge = pd.merge(ratings_merge, items, on='movie_id', how = 'left')
    ratings_merge = ratings_merge.drop(['occupation','zip_code','title',
                                        'release date','video release date', 'IMDb URL', 'unknown'], axis=1)

    ratings_merge = ratings_merge.replace('M',1)
    ratings_merge = ratings_merge.replace('F',-1)
    ones = np.ones(ratings_merge.shape[0])
    ratings_merge['bias'] = ones
    ratings_merge = ratings_merge.drop(['user_id', 'movie_id'], axis=1)
    train, test = train_test_split(ratings_merge, test_size=0.2)
    
    y_train = train['rating'].values
    x_train = train.drop(['rating'], axis=1).values
    y_test = test['rating'].values
    x_test = test.drop(['rating'], axis=1).values
    
    return x_train, y_train, x_test, y_test


# In[261]:


def analyze_and_plot_data(ratings):
    """
    Plots and prints some of the dataset information, including the number of users and movies, the
    distribution of ratings, the ratings per user and the ratings per movie.
    
    Input:  
    - ratings:  A pandas dataframe containing the ratings information.
    """
    
    num_users = ratings['user_id'].nunique()
    num_items = ratings['movie_id'].nunique()
    print("Number of unique users is " + str(num_users))
    print("Number of unique movies is " + str(num_items))
    print("The number of ratings in the dataset set is " + str(ratings.shape[0]))

    #Determine ratings distribution and plot results
    count = ratings['rating'].value_counts()
    count = count.to_frame('count')
    count.index.name = 'Rating'
    count = count.sort_values(by='Rating', ascending=1)
    count.plot(kind='bar')
    plt.ylabel('Number of ratings')
    plt.title('Distribution of Ratings')
    plt.savefig('ratings_distribution.png')

    #Pie plot
    count.plot(kind='pie', subplots=True, figsize=(5, 5), autopct='%1.0f%%')
    plt.title('Distribution of Ratings')
    plt.savefig('ratings_distribution_pie.png')
    plt.show()

    #Determine number of ratings per movie and plot data 
    count_movies_rated = ratings['movie_id'].value_counts()
    buckets = [250, 150, 50, 25, 5, 1]
    ratings_dist = np.zeros(6)
    prior_count = 0
    for i in range(6):
        ratings_dist[i] = count_movies_rated[count_movies_rated >= buckets[i]].count()
        ratings_dist[i] -= prior_count
        prior_count += ratings_dist[i]

    plt.title('Ratings per Movie')
    plt.xlabel('Number of ratings')
    plt.ylabel('Number of movies')
    label = ['>250','150-250', '50-150','50-25', '25-5', '1-5']
    index = np.arange(len(label))
    plt.bar(index, ratings_dist)
    plt.xticks(index, label)
    plt.savefig('movies_distribution.png')

    plt.show()

    #Determine how the number of ratings per user and plot data
    count_users = ratings['user_id'].value_counts()
    buckets = [250, 150, 50, 25, 5, 1]
    users_dist = np.zeros(6)
    prior_count = 0
    for i in range(6):
        users_dist[i] = count_users[count_users >= buckets[i]].count()
        users_dist[i] -= prior_count
        prior_count += users_dist[i]

    plt.title('Ratings per User')
    plt.xlabel('Number of ratings')
    plt.ylabel('Number of users')
    plt.bar(index, users_dist)
    plt.xticks(index, label)
    plt.savefig('users_distribution.png')

    plt.show()


# In[262]:


# Plot the training losses
def plot_loss(loss_history):
    """
    Helper function to plot the loss history of the neural network.  Used for diagnosing convergence.
    
    Input:
    - loss_history:  A numpy array containing the loss data.
    """
    plt.title('Loss history')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.plot(loss_history)
    plt.show()


# In[263]:


def get_accuracy(y_batch, y_pred):
    """
    Calculates and prints the accuracy for the given inputs.  Also, the accuracy for each rating is shown as well.
    The ratings have values between 1-5.
    
    Inputs:  
    y_batch:  A numpy array or a pandas series containing the true labels.
    y_pred:  A numpy array of predicted labels.
    """
    
    mse = mean_squared_error(y_batch, y_pred)
    print("The mean squared error is %.3f" % (mse))
    
    if isinstance(y_batch, pd.Series):
        y_batch = y_batch.values
    num_samples = y_batch.shape[0]
    y_pred = y_pred.round()
    y_pred[y_pred > 5] = 5
    num_correct = (y_pred == y_batch).sum()
    acc = float(num_correct) / num_samples
    print('Got %d / %d correct (%.2f%%)' % (num_correct, num_samples, 100 * acc))
    
    #Breakout accuracy by score
    correct = np.zeros(6)
    totals = np.zeros(6)
    for i in range(num_samples):
        totals[int(y_batch[i])] += 1
        if y_pred[i] == y_batch[i]:
            correct[y_batch[i]] += 1
    
    for i in range(1,6):
        print("Accuracy for ratings %d is %.2f%%" % (i, float(correct[i]/totals[i])*100))


# In[264]:


def calculate_averages(train, test, num_movies):
    """
    This is the simple method which is used as a baseline for comparison.  The predicted rating is calculated 
    as the average of the ratings in the training set.  
    
    Inputs:
    - train:  A pandas dataframe containing the ratings for the training set.
    - test:  A pandas datafrme containing the ratings for the test set.
    - num_movies:  the total number of movies in the dataset.  
    """
    
    N = num_movies + 1
    average_rating = np.zeros(N)
    for i in range(1, N): #(movie_ids start at 1)
        average_rating[i] = train[train['movie_id']==i].rating.mean()
    
    #clean up data for movies that didn't have any ratings
    index=0
    for i in average_rating[:]:
        if not np.isfinite(i):
            average_rating[index] = 0.0
        index +=1
    
    pred_ratings = np.zeros(test.shape[0])
    for i in range(test.shape[0]):
        pred_ratings[i] = average_rating[test['movie_id'].iloc[i]]
    print("Results for the average rating method:")
    get_accuracy(test['rating'], pred_ratings)
    print()


# In[265]:


def content_based(utility_matrix, items, test, k):
    """
    This is the method using content-based information from the movies.  The movies in the items dataframe
    are compared using a cosine similarity index.  The ratings of the top k movies that are similar to  
    each movie in the training set are weighted (according to the similarity index) and used as the prediction
    for the rating.
    
    Inputs:
    - utility_matrix:  An m x n numpy array, with m users and n items.  Each element represents the rating
      that that user m gives to movie n.  Each rating is between 1-5.  If the rating is unknown, the element is 0.
    - items:  A pandas dataframe containing the movie information.
    - test:  A pandas dataframe that represents the test set.
    - k:  A scalar, representing the top k number of items in the similarity index that will be used to generate the
      prediction.
      
    Returns:
    - pred_ratings:  A numpy array that contains the predicted ratings for the test set.  
    
    """
    
    np.seterr(divide='ignore', invalid='ignore')

    items_v2 = items.drop(['movie_id','title','release date','video release date', 'IMDb URL', 'unknown'], axis=1)
    item_sim = cosine_similarity(items_v2, items_v2)
    N = test.shape[0]
    pred_ratings = np.zeros(N)
    for i in range(N):
        movie = test['movie_id'].iloc[i] - 1
        top_k_movies = [np.argsort(item_sim[movie,:])[:-k-1:-1]]
        avg_ratings = np.true_divide(utility_matrix[:,top_k_movies].sum(axis=0), (utility_matrix[:,top_k_movies]!=0).sum(axis=0))
        top_sim_scores = item_sim[movie, top_k_movies]
        pred = avg_ratings.dot(top_sim_scores.T)
        nonzero = avg_ratings.nonzero()
        pred /= np.sum(np.abs(top_sim_scores[nonzero]))
    
        if math.isnan(pred):
            pred = 0.0
        pred_ratings[i] = pred

    print("Results for content-based method using the top " + str(k) + " items' data: ")
    get_accuracy(test['rating'], pred_ratings)
    print()
    
    return pred_ratings    


# In[266]:


def user_CF(utility_matrix, test, k):
    """
    The collaborative filtering method that predicts ratings based on users.  The similarity of the users in the
    utility matrix is computed.  The top k users to each user is then examined.  Their ratings for the movie in the 
    training set is weighted to predict the rating for the movie and user in the test set.  
    
    Inputs:
     - utility_matrix:  An m x n numpy array, with m users and n items.  Each element represents the rating
      that that user m gives to movie n.  Each rating is between 1-5.  If the rating is unknown, the element is 0.
    - test:  A pandas dataframe respresenting the test set. 
    - k:  A scalar, representing the top k number of users in the similarity index that will be used to generate the
      prediction.
      
    Returns:
    - pred_ratings:  A numpy array that contains the predicted ratings for the test set.  
    
    """
    
    user_sim = cosine_similarity(utility_matrix, utility_matrix)
    all_users = utility_matrix.shape[0]
    
    #Run using all of the users' data
    N = test.shape[0]
    pred_ratings = np.zeros(N)
    for i in range(N):
        user = test['user_id'].iloc[i] - 1
        movie = test['movie_id'].iloc[i] - 1
        top_k_users = [np.argsort(user_sim[user,:])[:-all_users-1:-1]]
        pred = user_sim[user,:][top_k_users].dot((utility_matrix[:,movie][top_k_users]).T)
        nonzero = utility_matrix[:,movie][top_k_users].nonzero()
        pred /= np.sum(np.abs(user_sim[user, :][top_k_users][nonzero]))
        if math.isnan(pred):
            pred = 0.0
        pred_ratings[i] = pred
    
    print("Results for user-based collaboritve filtering method using all of the users' data: ")
    get_accuracy(test['rating'], pred_ratings)
    print()
    
    #Run using only the top-k users' data
    for i in range(N):
        user = test['user_id'].iloc[i] - 1
        movie = test['movie_id'].iloc[i] - 1
        top_k_users = [np.argsort(user_sim[user,:])[:-k-1:-1]]
        pred = user_sim[user,:][top_k_users].dot((utility_matrix[:,movie][top_k_users]).T)
        nonzero = utility_matrix[:,movie][top_k_users].nonzero()
        pred /= np.sum(np.abs(user_sim[user, :][top_k_users][nonzero]))
        if math.isnan(pred):
            pred = 0.0
        pred_ratings[i] = pred
    print("Results for user-based collaboritve filtering method using the top " + str(k) + " users' data: ")
    get_accuracy(test['rating'], pred_ratings)
    print()
    
    return pred_ratings


# In[267]:


def item_CF(utility_matrix, test, k):
    """
    The collaborative filtering method that predicts ratings based on movies.  The similarity of the movies in the
    utility matrix is computed.  The top k movies to each movie is then examined.  Their ratings for that movie i
    in the training set is weighted to predict the rating for the movie and user in the test set.  
    
    Inputs:
     - utility_matrix:  An m x n numpy array, with m users and n items.  Each element represents the rating
      that that user m gives to movie n.  Each rating is between 1-5.  If the rating is unknown, the element is 0.
    - test:  A pandas dataframe respresenting the test set. 
    - k:  A scalar, representing the top k number of users in the similarity index that will be used to generate the
      prediction.
      
    Returns:
    - pred_ratings:  A numpy array that contains the predicted ratings for the test set.  
    
    """
    
    item_sim = cosine_similarity(utility_matrix.T, utility_matrix.T)
    all_items = utility_matrix.shape[1]

    #Run using all of the items' data
    N = test.shape[0]
    pred_ratings = np.zeros(N)
    for i in range(N):
        user = test['user_id'].iloc[i] - 1
        movie = test['movie_id'].iloc[i] - 1
        top_k_movies = [np.argsort(item_sim[movie,:])[:-all_items-1:-1]]
        pred = item_sim[movie,:][top_k_movies].dot((utility_matrix[user,:][top_k_movies]).T)
        nonzero = utility_matrix[user,:][top_k_movies].nonzero()
        pred /= np.sum(np.abs(item_sim[movie, :][top_k_movies][nonzero]))
        
        if math.isnan(pred):
            pred = 0.0
        pred_ratings[i] = pred
    

    print("Results for item-based collaboritve filtering method using all of the items' data: ")
    get_accuracy(test['rating'], pred_ratings)
    print()
    
    #Run using only the top-k items' data
    pred_ratings = np.zeros(N)
    for i in range(N):
        user = test['user_id'].iloc[i] - 1
        movie = test['movie_id'].iloc[i] - 1
        top_k_movies = [np.argsort(item_sim[movie,:])[:-k-1:-1]]
        pred = item_sim[movie,:][top_k_movies].dot((utility_matrix[user,:][top_k_movies]).T)
        nonzero = utility_matrix[user,:][top_k_movies].nonzero()
        pred /= np.sum(np.abs(item_sim[movie, :][top_k_movies][nonzero]))
        
        if math.isnan(pred):
            pred = 0.0
        pred_ratings[i] = pred
        
    print("Results for item-based collaboritve filtering method using the top " + str(k) + " items' data: ")
    get_accuracy(test['rating'], pred_ratings)
    print()
    
    return pred_ratings
    


# In[268]:


def SGD(ratings, utility_matrix, n_factors):
    """
    Stochastic gradient descent algorithm used for the matrix factorization method.
    
    Inputs:
    - ratings:  A pandas dataframe representing the ratings information
    - utility_matrix:  An m x n numpy array, with m users and n items.  Each element represents the rating
      that that user m gives to movie n.  Each rating is between 1-5.  If the rating is unknown, the element is 0.
    - n_factors:  A scalar representing the number of factors that will be calculated for each matrix.  Essentially, 
      it is the number of columns and number of rows in the p a q matrices respectively.
      
    Returns:
    -p,q,bi,bu:  p and q are matrics and bi and bu are numpy arrays for the matrix factorization method.  
    """
    
    alpha = .01  # learning rate
    gamma = .02
    n_epochs = 50  # number of iteration of the SGD procedure
    num_users = utility_matrix.shape[0]
    num_items = utility_matrix.shape[1]

    # Randomly initialize the user and item factors.
    p = np.random.normal(scale=1./n_factors, size=(num_users, n_factors))
    q = np.random.normal(scale=1./n_factors, size=(num_items, n_factors))
    bu = np.zeros(num_users)
    bi = np.zeros(num_items)
    
    loss_history = []
    # Optimization procedure
    for _ in range(n_epochs):
        for row in ratings.itertuples():
            u = row[1] - 1
            i = row[2] - 1
            err = row[3] - (np.dot(p[u],q[i])) #+bu[u]+bi[i]+b)
            loss_history.append(err)
            # Update vectors p_u and q_i
            p[u] += alpha * (err * q[i] - gamma*p[u])
            q[i] += alpha * (err * p[u] - gamma*q[i])
            #bu[u] += alpha * (err - gamma * bu[u])
            #bi[i] += alpha * (err - gamma * bi[i])
    
    plt.title('Loss history')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.plot(loss_history)
    plt.show()
    return p, q, bu, bi


# In[269]:


def matrix_factorization(utility_matrix, train, test, num_factors=10):
    """
    Matrix factorization algorithms work by decomposing the utility matrix into the product of 
    two lower-dimensionality rectangular matrices, M and U.  M and U are similar to the decomposition used
    in SVD.  These matrices are learned by finding matrices p and q.  p is size m x num_factors and q is 
    size n x num_factors.  These matrices are learned optimizing the function r - p * q (
    where r represents the known ratings) using stochastic gradient descent. 
    
    Inputs:
    - utility_matrix:  An m x n numpy array, with m users and n items.  Each element represents the rating
      that that user m gives to movie n.  Each rating is between 1-5.  If the rating is unknown, the element is 0.
    - train:  A pandas dataframe representing the training set
    - test:  A pandas dataframe representing the testing set
    - num_factors:  A scalar representing the number of factors that will be calculated for each matrix.  Essentially, 
      it is the number of columns in the p and q matrices.
    """
    
    p, q, bu, bi = SGD(train, utility_matrix, num_factors)
    N = test.shape[0]
    pred_ratings = np.zeros(N)
    ui_matrix = p.dot(q.T)
    for i in range(N):
        user = test['user_id'].iloc[i] - 1 
        item = test['movie_id'].iloc[i] - 1
        pred_ratings[i] = ui_matrix[user,item] + bi[item] + bu[user] 
    print("Results for matrix factorization method: ")
    get_accuracy(test['rating'], pred_ratings)
    print()
    
    return pred_ratings


# In[270]:


def hybrid(hybrid_ratings, test):
    """
    This method is a weighted combination of the collaborative filtering predictions and the matrix factorization
    predictions.  It is based on the fact that the collaborative filtering predictions are more accurate
    for ratings 3 and 4 while the matrix factorization is more accurate for ratings 1 and 5.  
    
    Inputs:
    - hybrid_ratings:  a list of numpy arrays that contains the predicted ratings for the two collaborative filtering
      methods and the matrix factorization method.
    - test:  A pandas dataframe that represents the test set.
    """
    
    N = test.shape[0]
    u_weights = [0, 0, 3, 12, 12, 0, 0]
    i_weights = [0, 1, 3, 8, 8, 0, 0]
    SVD_weights = [0, 20, 3, 1, 1, 20, 0]
    
    #round ratings for comparison purposes
    users = np.abs(hybrid_ratings[0].round())
    users[users > 5] = 5
    items = np.abs(hybrid_ratings[1].round())
    items[items > 5] = 5
    SVD = np.abs(hybrid_ratings[2].round())
    SVD[SVD > 5] = 5

    u_col = np.zeros(N)
    i_col = np.zeros(N)
    SVD_col = np.zeros(N)
    for i in range(N):
        u_col[i] = u_weights[int(users[i])] 
        i_col[i] = i_weights[int(items[i])]
        SVD_col[i] = SVD_weights[int(SVD[i])]
    
    pred_ratings = np.zeros(N)
    pred_ratings = (hybrid_ratings[0] * u_col + hybrid_ratings[1]*i_col + hybrid_ratings[2]*SVD_col) / (u_col + i_col + SVD_col)
    pred_ratings = np.nan_to_num(pred_ratings)

    print("Results for the hybrid method :")
    get_accuracy(test['rating'], pred_ratings)
    print()


# In[271]:


def check_accuracy_nn(sess, x_test, y_test, x, scores, is_training=None):
    """
    Check accuracy on a classification model.
    
    Inputs:
    - sess: A TensorFlow Session that will be used to run the graph
    - x_test: A numpy array containg the testing input
    - y_test:  A numpy array containg the true labels
    - x: A TensorFlow placeholder Tensor where input images should be fed
    - scores: A TensorFlow Tensor representing the scores output from the
      model; this is the Tensor we will ask TensorFlow to evaluate.
    - is_trainin:  A TensorFlow placeholder Boolean value that represents whether the model is training
      or predicting.
      
    Returns: Nothing, but prints the accuracy of the model
    """
    num_correct = 0
    num_samples = x_test.shape[0]
    feed_dict = {x: x_test,is_training: 0}        
    scores_np = sess.run(scores, feed_dict=feed_dict)
    y_pred = scores_np.argmax(axis=1)
    print("Results for the neural network: ")
    get_accuracy(y_test, y_pred)
    print()
    


# In[272]:


def model_init_fn(inputs, is_training):
    """
    The neural network model.  This model has three dense layers.
    
    """
    input_shape = ((22))
    hidden_layer_size, num_classes = 200, 6
    initializer = tf.variance_scaling_initializer(scale=1.0)
    layers = [
        tf.layers.Flatten(),
        tf.layers.Dense(hidden_layer_size, activation=tf.nn.tanh,
                        kernel_initializer=initializer),
        tf.layers.Dense(hidden_layer_size, activation=tf.nn.tanh,
                        kernel_initializer=initializer),
        tf.layers.Dense(hidden_layer_size, activation=tf.nn.tanh,
                        kernel_initializer=initializer),
        tf.layers.Dense(num_classes, kernel_initializer=initializer),
    ]
    model = tf.keras.Sequential(layers)
    return model(inputs)

def optimizer_init_fn():
    """
    The optimizer function for the model.
    The Adam Optimizer is used.
    """

    return tf.train.AdamOptimizer(learning_rate)

def train_part(model_init_fn, optimizer_init_fn, x_train, y_train, x_test, y_test, num_epochs=5, batch_size=64):
    """
    Simple training loop for use with models defined using tf.keras. It trains
    a model for one epoch on the CIFAR-10 training set and periodically checks
    accuracy on the CIFAR-10 validation set.
    
    Inputs:
    - model_init_fn: A function that takes no parameters; when called it
      constructs the model we want to train: model = model_init_fn()
    - optimizer_init_fn: A function which takes no parameters; when called it
      constructs the Optimizer object we will use to optimize the model:
      optimizer = optimizer_init_fn()
    - x_train:  A numpy array of size (N x D) that contains the input for the training data
    - y_train:  A numpy array of size (N, ) that contains the true labels for the training data
    - x_test:  A numpy array of size (N x D) that contains the input for the test data
    - y_test:  A numpy array of size (N, ) that contains the true labels for the test data
    - num_epochs: The number of epochs to train for
    - batch_size:  The batch size for cross-validation
    
    Returns: Nothing, but prints progress during trainingn
    """
    tf.reset_default_graph()    
    with tf.device('/cpu:0'):
        x = tf.placeholder(tf.float32, [None, 22])
        y = tf.placeholder(tf.int32, [None])
        is_training = tf.placeholder(tf.bool, name='is_training')
        scores = model_init_fn(x, is_training)
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=scores)
        loss = tf.reduce_mean(loss)
        optimizer = optimizer_init_fn()
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = optimizer.minimize(loss)

    N = x_train.shape[0]
    loss_history = []
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        t = 0
        for epoch in range(num_epochs):
            index = np.random.permutation(N)
            x_np = x_train[index]
            y_np = y_train[index]
        
            #Iterate through randomly shuffled training examples
            start = 0
            end = batch_size
        
            #Iterate through randomly shuffled training examples
            while(end < N):
                x_batch = x_np[start:end]
                y_batch = y_np[start:end]
                feed_dict = {x: x_batch, y: y_batch, is_training:1}
                loss_np, _ = sess.run([loss, train_op], feed_dict=feed_dict)
                loss_history.append(loss_np)
                start += batch_size
                end += batch_size
                t += 1
        plot_loss(loss_history)

        check_accuracy_nn(sess, x_test, y_test, x, scores, is_training=is_training)


# In[288]:


def test_run(users, items, ratings, train, test, k=50):
    """
    Represents a test run for all of the methods.  The results are printed for each method.
    
    Inputs:
    - users: A pandas dataframe representing the user information.
    - items:  A pandas dataframe representing the movie information.
    - ratings:  A pandas dataframe representing the ratings information
    - train:  A pandas dataframe representing the training set
    - test:  A pandas dataframe representing the test set.
    - k: A scalar representing the number of users/items that will be compared using the content-based
      and collaborative filtering methods.
    """
    
    num_users = ratings['user_id'].nunique()
    num_items = ratings['movie_id'].nunique()
    print("Number of unique users is " + str(num_users))
    print("Number of unique movies is " + str(num_items))
    
    #Create utility matrix that will be used for most of the methods
    utility_matrix = np.zeros((num_users, num_items))
    i = 0
    for row in train.itertuples():  
        i += 1
        if row[1]-1 > num_users or row[2]-1> num_items:
            print("Row number " + str(i) + " is the problem!" )
        utility_matrix[row[1]-1, row[2]-1] = row[3]
    
    #Determine sparcity
    sparsity = float(len(utility_matrix.nonzero()[0]))
    sparsity /= (utility_matrix.shape[0] * utility_matrix.shape[1])
    sparsity *= 100
    print('Sparsity: {:4.2f}%'.format(sparsity))
    print()
    
    hybrid_ratings = []
    #Method #1 - Average ratings
    calculate_averages(train, test, num_items)
    
    #Method #2 - Content-based prediction
    content_based(utility_matrix, items, test, k)

    #Method #3 - User-based colloborative filtering
    hybrid_ratings.append(user_CF(utility_matrix, test, k))

    #Method #4 - Item-based colloborative filtering
    hybrid_ratings.append(item_CF(utility_matrix, test, k))
    
    #Method #5 - SVD-inspired
    hybrid_ratings.append(matrix_factorization(utility_matrix, train, test, num_factors=10))
    
    #Method #6 - Hybrid Method
    hybrid(hybrid_ratings, test)
    
    #Method #7 - Neural network
    x_train, y_train, x_test, y_test = process_data_for_NN(ratings, users, items, threshold=100)
    train_part(model_init_fn, optimizer_init_fn, x_train, y_train, x_test, y_test)
    


# In[214]:


download_data("./")


# In[185]:


users, items, ratings = load_data("", data_size='small')
#users.head()


# In[186]:


#items.head()


# In[187]:


#ratings.head()


# In[280]:


analyze_and_plot_data(ratings)


# In[189]:


#split data into test and train data
train, test = train_test_split(ratings, test_size=0.2)


# In[286]:


#First test with raw data
learning_rate = 0.01
test_run(users, items, ratings, train, test, k=50)


# In[276]:


#Make Data more dense
def reduce_users_and_items(ratings, users, items, threshold=100):
    """
    Reduces the number of users and items to create a more dense utility matrix.  Both the movies and items
    dataframes are reduced to contain only users/movies with more than the threshold number of ratings.
    
    """

    user_counts = ratings['user_id'].value_counts()
    ratings_v2 = ratings.loc[ratings['user_id'].isin(user_counts[user_counts >= threshold].index)]
    users_v2 = users.loc[users['user_id'].isin(user_counts[user_counts >= threshold].index)]

    item_counts = ratings['movie_id'].value_counts()
    ratings_v2 = ratings_v2.loc[ratings['movie_id'].isin(item_counts[item_counts >= threshold].index)]
    items_v2 = items.loc[items['movie_id'].isin(item_counts[item_counts >= threshold].index)]

    analyze_and_plot_data(ratings_v2)


    new_movie_id = np.arange(1, items_v2.shape[0]+1)
    new_movie = pd.Series(new_movie_id).rename('new_id')
    items_v2 = items_v2.assign(new_id=new_movie.values)

    new_user_id = np.arange(1, users_v2.shape[0]+1)
    new_user = pd.Series(new_user_id).rename('new_id')
    users_v2 = users_v2.assign(new_id=new_user.values)

    for i in range(items.shape[0]+1):
        new_id = items_v2.loc[items_v2['movie_id'] == i]['new_id']
        if (len(new_id) != 0):
            new = new_id.values
            new_movie_ids = ratings_v2['movie_id'].mask(ratings_v2['movie_id']==i, new)
            ratings_v2 = ratings_v2.assign(movie_id = new_movie_ids)

    
    for u in range(users.shape[0]+1):
        new_id = users_v2.loc[users_v2['user_id'] == u]['new_id']
        if (len(new_id) != 0):
            new = new_id.values
            new_user_ids = ratings_v2['user_id'].mask(ratings_v2['user_id']==u, new)
            ratings_v2 = ratings_v2.assign(user_id =new_user_ids)


    train, test = train_test_split(ratings_v2, test_size=0.2)
    items_v2 = items_v2.drop(['movie_id'], axis=1)
    users_v2 = users_v2.drop(['user_id'], axis=1)
    items_v2.rename(columns = {'new_id':'movie_id'}, inplace = True)
    users_v2.rename(columns = {'new_id':'user_id'}, inplace = True)

    return items_v2, users_v2, ratings_v2, train, test
    


# In[287]:


threshold = 100
print("Let's make the data more dense.  We will only include users with more than " + str(threshold) + " ratings "
      + "and movies with more than " + str(threshold) + " ratings.")

new_items, new_users, new_ratings, train, test = reduce_users_and_items(ratings, users, items, threshold=threshold)

print("All the tests will be run on the reduced dataset.")
print()
test_run(new_users, new_items, new_ratings, train, test, k=50)



