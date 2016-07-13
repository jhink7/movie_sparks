from flask import Blueprint
from time import time

rec_engine_app = Blueprint('rec_engine_app', __name__)

import json
from data_science.rec_engine import RecommendationEngine

import logging

logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)

from flask import Flask, request, jsonify, abort, make_response

@rec_engine_app.errorhandler(400)
def not_found(error):
    return make_response(jsonify({'error': 'Bad Request'}), 400)


@rec_engine_app.errorhandler(500)
def not_found(error):
    return make_response(jsonify({'error': 'Server Error'}), 500)


@rec_engine_app.route("/<int:user_id>/ratings/<int:movie_id>", methods=["GET"])
def movie_ratings(user_id, movie_id):
    try:
        rating = recommendation_engine.get_movie_rating_by_user(user_id, [movie_id])
        return jsonify({'rating': rating})
    except Exception as ex:
        abort(500)


@rec_engine_app.route("/<int:user_id>/ratings/top", methods=["GET"])
def top_ratings(user_id):
    try:
        recs = recommendation_engine.get_top_ratings_by_user(user_id, 10)
        return jsonify({'recs': recs})
    except Exception as ex:
        abort(500)


@rec_engine_app.route("/<int:user_id>/ratings", methods=["POST"])
def add_ratings(user_id):
    # get the ratings from the Flask POST request object
    ratings_list = request.form.keys()[0].strip().split("\n")
    ratings_list = map(lambda x: x.split(","), ratings_list)
    # create a list with the format required by the rec engine (user_id, movie_id, rating)
    ratings = map(lambda x: (user_id, int(x[0]), float(x[1])), ratings_list)
    # add them to the model using then engine API
    recommendation_engine.add_ratings(ratings)

    return json.dumps(ratings)

@rec_engine_app.route("/engine/reload-and-retrain", methods=["POST"])
def reload_retrain():
    try:
        post_data = request.get_json()

        if ('rank' in post_data) and ('seed' in post_data) and ('num_iterations' in post_data) and 'reg' in post_data:
            rank = post_data['rank']
            seed = post_data['seed']
            num_iterations = post_data['num_iterations']
            reg = post_data['reg']

            t0 = time()
            recommendation_engine.reload_and_retrain(rank, seed, num_iterations, reg)
            train_time = time() - t0
            return jsonify({'retrained': True, 'trainingTime': train_time})

    except Exception as ex:
        if "Bad Request" in str(ex):
            abort(400)
        else:
            logger.error(str(ex))
            abort(500)


def create_app(spark_context, dataset_path):
    global recommendation_engine

    recommendation_engine = RecommendationEngine(spark_context, dataset_path)

    app = Flask(__name__)
    app.register_blueprint(rec_engine_app)
    return app