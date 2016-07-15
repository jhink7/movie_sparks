from math import*
import pandas as pd
import numpy as np
import random as rd
from pyspark import SparkContext, SparkConf

from data_science.rec_engine import RecommendationEngine

data_root = 'data/toy'
conf = SparkConf().setAppName("movie_recommendation-eval")
# IMPORTANT: pass aditional Python modules to each worker
sc = SparkContext(conf=conf, pyFiles=['data_science/rec_engine.py'])

logger = sc._jvm.org.apache.log4j
logger.LogManager.getLogger("org").setLevel(logger.Level.ERROR)
logger.LogManager.getLogger("akka").setLevel(logger.Level.ERROR)

engine = RecommendationEngine(sc, data_root, False)

eval_ratings = pd.read_csv('data/toy/ratings_eval.csv')
m1_preds = []

for eval in eval_ratings.values:
    m1_hat = engine.get_movie_rating_by_user(eval[0], [eval[1]])
    m1_preds.append(round(m1_hat[0][1], 2))


eval_ratings['m1_hat'] = pd.Series(np.asarray(m1_preds), index=eval_ratings.index)

eval_ratings.to_csv('data/out/evals_dp_false.csv', index=False)

print m1_preds



