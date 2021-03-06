import os
import numpy as np
from pyspark.mllib.recommendation import ALS

import logging

logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)


def get_movie_summary_metrics(ratings):
    # Aggregate average ratings and counts for the given movie/rating input pair
    num_ratings = len(ratings[1])
    return ratings[0], (num_ratings, float(sum(x for x in ratings[1])) / num_ratings)


class RecommendationEngine:

    def __summarize_ratings(self, ratings_RDD):
        # Update counts for current data
        movie_ID_with_ratings_RDD = ratings_RDD.map(lambda x: (x[1], x[2])).groupByKey()
        movie_ID_with_avg_ratings_RDD = movie_ID_with_ratings_RDD.map(get_movie_summary_metrics)
        movies_rating_counts_RDD = movie_ID_with_avg_ratings_RDD.map(lambda x: (x[0], x[1][0]))
        return movies_rating_counts_RDD

    def __predict_ratings(self, user_and_movie_RDD):
        MAX_RATING = 5.0

        # Gets predicted ratings for a userid, movieid combo
        pred_rdd = self.model.predictAll(user_and_movie_RDD)

        # if differential privacy is enabled in the app, we'll add noise to our ratings
        # outputs will be stochastic
        if self.use_diff_priv:
            noise = 0.1
            pred_rating_RDD = pred_rdd.map(lambda x: (x.product, min(np.random.normal(x.rating, noise), MAX_RATING)))
        else:
            pred_rating_RDD = pred_rdd.map(lambda x: (x.product, min(x.rating, MAX_RATING)))
        predicted_rating_title_and_count_RDD = \
            pred_rating_RDD.join(self.movies_titles_RDD).join(self.movies_rating_counts_RDD)
        predicted_rating_title_and_count_RDD = \
            predicted_rating_title_and_count_RDD.map(lambda r: (r[1][0][1], r[1][0][0], r[1][1]))

        return predicted_rating_title_and_count_RDD

    def add_ratings(self, ratings):
        # Add new movie rating to the system
        new_ratings_RDD = self.sc.parallelize(ratings)
        self.ratings_RDD = self.ratings_RDD.union(new_ratings_RDD)
        self.movies_rating_counts_RDD = self.__summarize_ratings(self.ratings_RDD)
        self.model = self.__train_model(self.rank, self.seed, self.iterations, self.reg)
        return True

    def get_movie_rating_by_user(self, user_id, movie_ids):
        requested_movies_RDD = self.sc.parallelize(movie_ids)\
                                    .map(lambda x: (user_id, x))
        # Get predicted ratings
        ratings = self.__predict_ratings(requested_movies_RDD).collect()

        return ratings

    def get_top_ratings_by_user(self, user_id):

        # required ratings count for movie to be included in our algo
        min_rats_required = 12

        # number of top recommendations to return.
        # TODO:  Make this a configurable value
        num_recs = 5

        # gather only movies that have not been rated (seen) by the current user
        # note this assumes that users have rated every movie they've seen.  This is obviously flimsy
        # A separate list of movies viewed by the users could/should be used to filter here
        user_unrated_movies_rdd = self.ratings_RDD\
            .filter(lambda rat: not rat[0] == user_id) \
            .map(lambda r: (user_id, r[1]))\
            .distinct()

        # Get predicted ratings
        rat_hat = self.__predict_ratings(user_unrated_movies_rdd)\
            .filter(lambda rat: rat[2] >= min_rats_required)\
            .takeOrdered(num_recs, key=lambda r: -r[1])

        return rat_hat

    def __load_data(self):
        # load ratings data
        logger.info("Loading Data...")
        rat_path = os.path.join(self.data_root, 'ratings.csv')
        print rat_path
        rat_raw_RDD = self.sc.textFile(rat_path)
        header_rat = rat_raw_RDD.take(1)[0]
        ratings_RDD = rat_raw_RDD.filter(lambda line: line != header_rat) \
            .map(lambda line: line.split(","))\
            .map(lambda seg: (int(seg[0]), int(seg[1]), float(seg[2]))).cache()

        # load movies data
        movie_path = os.path.join(self.data_root, 'movies.csv')
        movies_raw_RDD = self.sc.textFile(movie_path)
        header_movie = movies_raw_RDD.take(1)[0]
        movies_RDD = movies_raw_RDD.filter(lambda line: line != header_movie) \
            .map(lambda line: line.split(",")).map(lambda m: (int(m[0]), m[1])).cache()
        movies_titles_RDD = movies_RDD.map(lambda m: (int(m[0]), m[1])).cache()

        return ratings_RDD, movies_RDD, movies_titles_RDD

    def reload_and_retrain(self, rank=8, seed=5L, num_iterations=15, reg=0.1):
        try:
            # Load intial training data
            ratings_RDD, movies_RDD, movies_titles_RDD = self.__load_data()
            self.movies_rating_counts_RDD = self.__summarize_ratings(ratings_RDD)
            # Train the intial model
            model = self.__train_model(rank, seed, num_iterations, reg)

            self.ratings_RDD = ratings_RDD
            self.movies_RDD = movies_RDD
            self.movies_titles_RDD = movies_titles_RDD
            self.model = model
        except Exception:
            logger.error("Error reloading data")
            raise Exception("Error reloading data")


    def __train_model(self, rank, seed, iterations, reg):
        logger.info("Training Movie Rec Engine (ALS)...")
        model = ALS.train(self.ratings_RDD, rank, seed=seed,
                               iterations=iterations, lambda_=reg)
        logger.info("Movie Rec Engine Trained!")

        self.rank = rank
        self.seed = seed
        self.iterations = iterations
        self.reg = reg

        return model

    def __init__(self, sc, data_root, use_diff_priv):

        # set the spark contex to that created at the web server layer
        self.sc = sc

        # set initial tweakable ALS parameters
        rank = 8
        seed = 5L
        num_iterations = 10
        reg = 0.16
        self.use_diff_priv = use_diff_priv
        self.data_root = data_root

        self.ratings_RDD, self.movies_RDD, self.movies_titles_RDD = self.__load_data()
        self.movies_rating_counts_RDD = self.__summarize_ratings(self.ratings_RDD)
        # trigger the training of our actual model
        self.model = self.__train_model(rank, seed, num_iterations, reg)