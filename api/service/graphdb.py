from platform import node

import networkx as nx
import networkx.algorithms.community as nx_comm
import json

from matplotlib import pyplot as plt
from numpy import sort


class GraphDB:

    def formatGraph(self, route):
        print("Loading JSON file...")
        data = json.load(open(route, 'r', encoding='utf-8'))
        print("Formatting JSON for NetworkX...")
        nodes = []
        edges = []
        self.read_all_nodes(data, nodes, edges)
        print("Building graph with " + str(len(nodes)) + " nodes and " + str(len(edges)) + " edges...")
        formatted_data = {'nodes': nodes, 'links': edges}
        with open('networkx-data.json', 'w') as outfile:
            json.dump(formatted_data, outfile)

    def loadGraph(self):
        print("Loading JSON file...")
        formatted_data = json.load(open('networkx-data.json', 'r', encoding='utf-8'))
        print("Building graph...")
        graph = nx.node_link_graph(formatted_data, directed=True, multigraph=False)
        print("Graph built!")
        return graph

    def loadGraph(self, networkFile):
        print("Loading JSON file...")
        formatted_data = json.load(open(networkFile, 'r', encoding='utf-8'))
        print("Building graph...")
        graph = nx.node_link_graph(formatted_data, directed=True, multigraph=False)
        print("Graph built!")
        return graph

    def getAllApps(self, graph):
        apps = []
        for node in graph.nodes:
            if 'https://schema.org/MobileApplication/' in node:
                apps.append(node)
        return apps

    # COMMUNITY DETECTION OPERATIONS
    def computeCommunities(self, graph, n, algorithm):
        graph = self.subgraphWithFeatures(graph)
        if algorithm == 'louvain':
            communities = nx_comm.louvain_communities(graph, seed=123)
        elif algorithm == 'modularity':
            if n is not None:
                communities = nx_comm.greedy_modularity_communities(graph, best_n=n)
            else:
                communities = nx_comm.greedy_modularity_communities(graph)
        else:
            if algorithm == 'girvan_newman':
                communities = nx_comm.girvan_newman(graph)
            elif algorithm == 'asyn_lpa':
                communities = nx_comm.asyn_lpa_communities(graph)
            elif algorithm == 'lpa':
                communities = nx_comm.label_propagation_communities(graph.to_undirected())
            communities = list(c for c in communities)

        print("Found " + str(len(communities)) + " communities")
        for x in range(0, len(communities)):
            communities[x] = list(communities[x])
        return communities

    def subgraphWithFeatures(self, graph):
        nodes = []
        for node in graph.nodes():
            if 'https://schema.org/DefinedTerm' in node:
                nodes.append(node)
        return graph.subgraph(nodes)

    # SIMILARITY OPERATIONS
    def computeTopKSimilarApps(self, graph, app_a, k=20, level=2, algorithm='simrank'):
        apps = self.getAllApps(graph)

        res = {}
        if algorithm == 'simrank':
            for app in app_a:

                rank = []
                for compare_app in apps:
                    score = self.computeSimilarityBetweenTwoApps(graph, app, compare_app, level)
                    rank.append({"documentID": compare_app, 'score': score,
                                 'category': graph.nodes[compare_app]['https://schema.org/applicationCategory']})

                sorted_rank = sorted(rank, key=lambda x: x['score'], reverse=True)
                res[app] = sorted_rank[0:k]
        elif algorithm == 'simrank-star':
            simrank_star_graph = nx.DiGraph()
            for app in apps:
                simrank_star_graph.add_node(app)
                neighbor_features = set(self.getNeighborFeatures(graph, app, level))
                subgraph = graph.subgraph(neighbor_features)
                for feature in neighbor_features:
                    simrank_star_graph.add_node(feature)
                    simrank_star_graph.add_edge(app, feature)
                    for edge in subgraph.edges:
                        simrank_star_graph.add_edge(edge[0], edge[1])
            simrank_star(simrank_star_graph, 10, k)

        return res

    def computeTopKAppsByFeature(self, graph, app_a, k=20, level=2, algorithm='simrank'):
        apps = self.getAllApps(graph)

        res = {}
        if algorithm == 'simrank':
            for app in app_a:

                rank = []
                for compare_app in apps:
                    score = self.computeSimilarityBetweenFeatureAndApp(graph, app, compare_app, level)
                    rank.append({"documentID": compare_app, 'score': score,
                                 'category': graph.nodes[compare_app]['https://schema.org/applicationCategory']})

                sorted_rank = sorted(rank, key=lambda x: x['score'], reverse=True)
                res[app] = sorted_rank[0:k]
        elif algorithm == 'simrank-star':
            # TODO
            print("TO-DO")

        return res

    def computeSimilarityBetweenTwoApps(self, graph, app_a, app_b, level):
        return self.simrankFeatureBased(graph, app_a, app_b, level)

    def computeSimilarityBetweenFeatureAndApp(self, graph, feature, app, level):
        return self.simrankFeatureBased(graph, feature, app, level)

    def simrankFeatureBased(self, graph, app_a, app_b, level):
        neighbor_features_a = set(self.getNeighborFeatures(graph, app_a, level))
        neighbor_features_b = set(self.getNeighborFeatures(graph, app_b, level))

        intersec = len(set(neighbor_features_a) & set(neighbor_features_b))
        return intersec

    def getNeighborFeatures(self, graph, app, level):
        direct_features = []
        for document in graph.neighbors(app):
            if 'https://schema.org/DigitalDocument' in document:
                for feature in graph.neighbors(document):
                    if 'https://schema.org/DefinedTerm' in feature:
                        direct_features.append(feature)
            # annotated features
            if 'https://schema.org/DefinedTerm' in document:
                direct_features.append(document)

        return self.getNeighborFeaturesByLevel(graph, direct_features, level)

    def getNeighborFeaturesByLevel(self, graph, features, k):
        if k == 1:
            return features
        else:
            neighbors = []
            for feature in features:
                for featureNeighbor in graph.neighbors(feature):
                    if 'https://schema.org/DefinedTerm' in featureNeighbor:
                        if feature not in neighbors:
                            neighbors.append(featureNeighbor)
            return features + self.getNeighborFeaturesByLevel(graph, neighbors, k - 1)

    def loadGraphForCommunityDetection(self):
        print("Loading JSON file...")
        formatted_data = json.load(open('networkx-data.json', 'r', encoding='utf-8'))
        print("Building graph...")
        graph = nx.node_link_graph(formatted_data, directed=True, multigraph=False)
        print("Graph built!")
        print("Proof-of-Concept communities report")

    def get_subgraph_nodes(self, graph, app):
        nodes_subgraph = [app]
        for neighbor in graph.neighbors(app):
            # if 'https://schema.org/Review' not in neighbor :
            nodes_subgraph.append(neighbor)
            for neighbor_2 in graph.neighbors(neighbor):
                nodes_subgraph.append(neighbor_2)
                for neighbor_3 in graph.neighbors(neighbor_2):
                    nodes_subgraph.append(neighbor_3)
        return nodes_subgraph

    def read_all_nodes(self, data, nodes, edges):
        for nodeId in data.keys():
            new_node = {"id": nodeId}
            if "https://schema.org/Review" not in nodeId and "https://schema.org/Person" not in nodeId:
                for property in data[nodeId].keys():

                    for data_item in data[nodeId][property]:
                        value = data_item['value']
                        if data_item['type'] == 'uri' and "https://schema.org/Review" not in value:
                            edge = {'source': nodeId, 'target': value, 'value': property}
                            edges.append(edge)
                        else:
                            if property in new_node.keys():
                                if isinstance(new_node[property], list):
                                    new_node[property].append(value)
                                else:
                                    new_node[property] = [new_node[property], value]
                            else:
                                new_node[property] = value

                nodes.append(new_node)

    def get_neighbors_visualization(self, graph, relevant_node, depth, file_name):
        nn = find_nn(graph, relevant_node, depth)
        edges = nx.edges(graph, nn)

        graph = nx.DiGraph()
        graph.add_nodes_from(nn)
        graph.add_edges_from(edges)

        f = plt.figure(figsize=(19.8, 10.8))
        nx.draw_networkx(graph)

        f.savefig(file_name)
        plt.clf()


'''
Based on: https://github.com/mrhhyu/EMB_vs_LB
by @masoud
'''
from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize
import numpy as np


def simrank_star(graph, iterations=10, topK=20):
    decay_factor = 0.8
    node_set = set()
    rows = []
    cols = []
    sign = []

    map_nodes = {}
    counter = 0

    for edge in graph.edges:

        row = -1
        column = -1

        # Map node ID to unique value
        if edge[0] not in map_nodes.keys():
            map_nodes[edge[0]] = counter
            row = counter
            counter += 1
        else:
            row = map_nodes[edge[0]]

        # Map node ID to unique value
        if edge[1] not in map_nodes.keys():
            map_nodes[edge[1]] = counter
            column = counter
            counter += 1
        else:
            column = map_nodes[edge[1]]

        rows.append(row)
        cols.append(column)
        sign.append(float(1))
        node_set.update((row, column))

    with open("map.json", "w") as outfile:
        json.dump(map_nodes, outfile)

    csr_adj = csr_matrix((sign, (rows, cols)), shape=(
    len(node_set), len(node_set)))  ## --- compressed sparse row representation of adjacency matrix

    print('The adjacency matrix is compressed in row format ...')

    norm_csr_adj = normalize(csr_adj, norm='l1', axis=0)
    print('Column normalization is done ...')

    iden_matrix = np.identity(len(node_set), dtype=float)
    del node_set
    del csr_adj
    iden_matrix = iden_matrix * (1 - decay_factor)
    result_ = iden_matrix  ## S_0

    for itr in range(1, iterations + 1):
        print("Iteration {} .... ".format(itr))
        result_ = (decay_factor / 2.0) * ((result_ @ norm_csr_adj).transpose() + result_ @ norm_csr_adj) + iden_matrix
        write_to_file(result_, topK, itr, map_nodes)


def write_to_file(result_matrix, topK, itr, map_nodes, type='/MobileApplication/'):
    '''
        Writes the results of each iteration in a file.
    '''
    sim_file = open('SRS_Top' + str(topK) + '_IT_' + str(itr), 'w')

    for target_node in range(0, len(result_matrix)):
        key_target_node = [k for k, v in map_nodes.items() if v == target_node][0]
        if type in key_target_node:
            target_node_res = result_matrix[target_node].tolist()
            # target_node_res_sorted = sorted(target_node_res,reverse=True)[:topK+1]
            target_node_res_sorted = sorted(target_node_res, reverse=True)[:topK + 1]

            for index in range(0, len(target_node_res_sorted)):
                val = target_node_res_sorted[index]
                if val != 0 and target_node_res.index(val) != target_node:
                    sim_file.write(
                        str(target_node) + ',' + str(target_node_res.index(val)) + ',' + str(round(val, 5)) + '\n')
                target_node_res[target_node_res.index(val)] = np.nan

    sim_file.close()
    print("The result of SimRank*, iteration {} is written in the file!.".format(itr))
    print('=============================================================================')


def find_nn_rec(G, current_node, k, curr_depth):
    if curr_depth >= k:
        return [current_node]
    else:
        current_neighbours = nx.neighbors(G, current_node)
        to_return = []
        for neighbor in current_neighbours:
            new_neighbours = find_nn_rec(G, neighbor, k, curr_depth + 1)
            to_return.extend(new_neighbours)
        return to_return


def find_nn(G, initial_node, k):
    return find_nn_rec(G, initial_node, k - 1, 0)
