from http import HTTPStatus
import flask
from flask import Blueprint, request
from flasgger import swag_from
from api.service.community_detection_service import CommunityDetectionService
from api.service.similarity_service import SimilarityService
from api.service.graphdb import GraphDB

home_api = Blueprint('api', __name__)

@home_api.route('/')
def welcome():
    return "OK", 200

@home_api.route('/graph')
def formatGraph():
    graphDB = GraphDB()
    graphDB.formatGraph(route = "C:\\Users\\QuimMotger\\AndroidStudioProjects\\app_data_repository\\src\\main\\resources\\statements.rj")
    return "OK", 200

@home_api.route('/computeSimilarity', methods = ['POST'])
def computeSimilarity():
    app_a = request.get_json()
    app_b = request.args.get('app_b')

    k = request.args.get('k', type=int)
    if k is None:
        k = 50

    level = request.args.get('level', type=int)
    if level is None:
        level = 2

    prefix = 'https://schema.org/MobileApplication/'

    graphDB = GraphDB()
    graph = graphDB.loadGraph()

    for i in range(0, len(app_a)):
        app_a[i] = prefix + app_a[i]
    
    if app_b is None:
        return graphDB.computeTopKSimilarApps(graph, app_a, k, level), 200
    else:
        app_b = prefix + app_b
        return graphDB.computeSimilarityBetweenTwoApps(graph, app_a, app_b, level), 200