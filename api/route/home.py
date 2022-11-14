from http import HTTPStatus
import flask, json
from flask import Blueprint, request
from flasgger import swag_from
from api.service.graphdb import GraphDB

home_api = Blueprint('api', __name__)


@home_api.route('/')
def welcome():
    return "OK", 200


@home_api.route('/graph')
def formatGraph():
    graphDB = GraphDB()
    route = request.args.get('route')
    graphDB.formatGraph(route=route)
    return "OK", 200


@home_api.route('/computeCommunities', methods=['POST'])
def computeCommunities():
    k = request.args.get('k', type=int)

    networkFile = request.args.get('network-file')
    algorithm = request.args.get('algorithm')
    if algorithm is None:
        algorithm = 'modularity'

    graphDB = GraphDB()
    if networkFile is not None:
        graph = graphDB.loadGraph(networkFile)
    else:
        graph = graphDB.loadGraph()

    return json.dumps(graphDB.computeCommunities(graph, k, algorithm)), 200


@home_api.route('/getTopKSimilarApps', methods=['POST'])
def getTopKSimilarApps():
    app_a = request.get_json()
    # app_b = request.args.get('app_b')
    algorithm = request.args.get('algorithm')
    k = request.args.get('k', type=int)
    level = request.args.get('level', type=int)
    networkFile = request.args.get('network-file')

    prefix = 'https://schema.org/MobileApplication/'

    graphDB = GraphDB()
    if networkFile is not None:
        graph = graphDB.loadGraph(networkFile)
    else:
        graph = graphDB.loadGraph()

    for i in range(0, len(app_a)):
        app_a[i] = prefix + app_a[i]

    # if app_b is None:
    return graphDB.computeTopKSimilarApps(graph, app_a, k, level, algorithm), 200
    # else:
    #    app_b = prefix + app_b
    #    return graphDB.computeSimilarityBetweenTwoApps(graph, app_a, app_b, level), 200


@home_api.route('/getTopKAppsByFeature', methods=['POST'])
def getTopKAppsByFeature():
    features = request.get_json()
    algorithm = request.args.get('algorithm')
    k = request.args.get('k', type=int)
    level = request.args.get('level', type=int)
    networkFile = request.args.get('network-file')

    prefix = 'https://schema.org/DefinedTerm/'

    graphDB = GraphDB()
    if networkFile is not None:
        graph = graphDB.loadGraph(networkFile)
    else:
        graph = graphDB.loadGraph()

    for i in range(0, len(features)):
        features[i] = prefix + features[i].replace(" ", "")

    return graphDB.computeTopKAppsByFeature(graph, features, k, level, algorithm), 200


@home_api.route('/visualize-node')
def visualizeNode():
    graphdb = GraphDB()
    node = request.args.get('node')
    route = request.args.get('route')
    if not route:
        route = 'test-api.png'
    graph = graphdb.loadGraph('networkx-data.json')
    if node:
        graphdb.get_neighbors_visualization(graph, relevant_node=node, depth=1, file_name=route)
        return 'OK', 200
    else:
        return 'Bad Request', 400
