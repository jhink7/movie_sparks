import time, sys, os
import cherrypy as cp
from paste.translogger import TransLogger
#import paste
from app import create_app
from pyspark import SparkContext, SparkConf


def init_spark_context():
    # load spark context
    conf = SparkConf().setAppName("movie_recommendation-server")
    # IMPORTANT: pass aditional Python modules to each worker
    sc = SparkContext(conf=conf, pyFiles=['data_science/rec_engine.py', 'app.py'])

    logger = sc._jvm.org.apache.log4j
    logger.LogManager.getLogger("org").setLevel(logger.Level.ERROR)
    logger.LogManager.getLogger("akka").setLevel(logger.Level.ERROR)

    return sc


def run_server(app):
    # Enable WSGI access logging via Paste
    app_logged = TransLogger(app)

    # Mount the WSGI callable object (app) on the root directory
    cp.tree.graft(app_logged, '/')

    # Start the CherryPy WSGI web server
    cp.engine.start()
    cp.engine.block()


if __name__ == "__main__":
    # Init spark context and load libraries
    sc = init_spark_context()
    #data_root = os.path.join('data/extracted', 'ml-latest-small')
    data_root = 'data/extracted/ml-latest-small'
    app = create_app(sc, data_root)

    # start web server
    run_server(app)