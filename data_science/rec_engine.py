import os
from pyspark.mllib.recommendation import ALS

import logging

logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)


def get_movie_summary_metrics(ratings):
    # Aggregate average ratings and counts for the given movie/rating input pair
    num_ratings = len(ratings[1])
    return ratings[0], (num_ratings, float(sum(x for x in ratings[1])) / num_ratings)


class RecommendationEngine:

    def __summarize_ratings(self):
        # Update counts for current data
        movie_ID_with_ratings_RDD = self.ratings_RDD.map(lambda x: (x[1], x[2])).groupByKey()
        movie_ID_with_avg_ratings_RDD = movie_ID_with_ratings_RDD.map(get_movie_summary_metrics)
        self.movies_rating_counts_RDD = movie_ID_with_avg_ratings_RDD.map(lambda x: (x[0], x[1][0]))

    def __predict_ratings(self, user_and_movie_RDD):

        # Gets predicted ratings for a userid, movieid combo
        pred_rdd = self.model.predictAll(user_and_movie_RDD)
        pred_rating_RDD = pred_rdd.map(lambda x: (x.product, x.rating))
        predicted_rating_title_and_count_RDD = \
            pred_rating_RDD.join(self.movies_titles_RDD).join(self.movies_rating_counts_RDD)
        predicted_rating_title_and_count_RDD = \
            predicted_rating_title_and_count_RDD.map(lambda r: (r[1][0][1], r[1][0][0], r[1][1]))

        return predicted_rating_title_and_count_RDD

    def add_ratings(self, ratings):
        # Add new movie rating to the system
        new_ratings_RDD = self.sc.parallelize(ratings)
        self.ratings_RDD = self.ratings_RDD.union(new_ratings_RDD)
        self.__summarize_ratings()
        self.__train_model()

        return ratings

    def get_movie_rating_by_user(self, user_id, movie_ids):
        requested_movies_RDD = self.sc.parallelize(movie_ids)\
                                    .map(lambda x: (user_id, x))
        # Get predicted ratings
        ratings = self.__predict_ratings(requested_movies_RDD).collect()

        return ratings

    def get_top_ratings_by_user(self, user_id, num_recs):

        # required ratings count for movie to be included in our algo
        min_rats_required = 10

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

    def __load_data(self, path):
        # load ratings data
        logger.info("Loading Data...")
        rat_path = os.path.join(path, 'ratings.csv')
        raw_RDD = self.sc.textFile(rat_path)
        header_rat = raw_RDD.take(1)[0]
        self.ratings_RDD = raw_RDD.filter(lambda line: line != header_rat) \
            .map(lambda line: line.split(",")).map(
            lambda tokens: (int(tokens[0]), int(tokens[1]), float(tokens[2]))).cache()

        # load movies data for later use
        movie_path = os.path.join(path, 'movies.csv')
        movies_raw_RDD = self.sc.textFile(movie_path)
        header_movie = movies_raw_RDD.take(1)[0]
        self.movies_RDD = movies_raw_RDD.filter(lambda line: line != header_movie) \
            .map(lambda line: line.split(",")).map(lambda m: (int(m[0]), m[1], m[2])).cache()
        self.movies_titles_RDD = self.movies_RDD.map(lambda m: (int(m[0]), m[1])).cache()


    def __train_model(self):
        logger.info("Training Movie Rec Engine (ALS)...")
        self.model = ALS.train(self.ratings_RDD, self.rank, seed=self.seed,
                               iterations=self.iterations, lambda_=self.regularization_parameter)
        logger.info("Movie Rec Engine Trained!")

    def __init__(self, sc, dataset_path):

        # set the spark contex to that created at the web server layer
        self.sc = sc

        # set tweakable ALS parameters
        self.rank = 8
        self.seed = 5L
        self.iterations = 15
        self.regularization_parameter = 0.1

        self.__load_data(dataset_path)
        self.__summarize_ratings()
        # trigger the training of our actual model
        self.__train_model()