from platform import node
import networkx as nx
import json
from numpy import sort

from sklearn import neighbors

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

    def getAllApps(self, graph):
        apps = []
        for node in graph.nodes:
            if 'https://schema.org/MobileApplication/' in node:
                apps.append(node)
        return apps

    def computeTopKSimilarApps(self, graph, app_a, k, level):
        apps = self.getAllApps(graph)

        rank = []

        for compare_app in apps:
            score = self.computeSimilarityBetweenTwoApps(graph, app_a, compare_app, level)
            rank.append({"documentID": compare_app, 'score': self.computeSimilarityBetweenTwoApps(graph, app_a, compare_app, level), 'category': graph.nodes[compare_app]['https://schema.org/applicationCategory']})
            #rank[compare_app] = self.computeSimilarityBetweenTwoApps(graph, app_a, compare_app, level)

        #TODO transform this so that the output is the same as index in GraphDB
        # then check if possible to improve edge generation
        sorted_rank = sorted(rank, key=lambda x:x['score'], reverse=True)

        return sorted_rank[0:k]
        
    def computeSimilarityBetweenTwoApps(self, graph, app_a, app_b, level):
       
        #print("Proof-of-Concept similarity report")
        # SubGraph1
        nodes_subgraph_a = self.get_subgraph_nodes(graph, app_a)
        nodes_subgraph_b = self.get_subgraph_nodes(graph, app_b)

        return self.simrankFeatureBased(graph, app_a, app_b, level)
        #self.simrankFeatureBased(graph, app_a, app_c)

    def simrankFeatureBased(self, graph, app_a, app_b, level):
        neighbor_features_a = set(self.getNeighborFeatures(graph, app_a, level))
        neighbor_features_b = set(self.getNeighborFeatures(graph, app_b, level))
        #scale = 1 / (len(neighbor_features_a) * len(neighbor_features_b))

        #print("Features in " + app_a)
        #for x in neighbor_features_a:
        #    print(x)
        #print("")
        #print("Features in " + app_b)
        #for x in neighbor_features_b:
        #    print(x)
        #print("")
        intersec = len(set(neighbor_features_a) & set(neighbor_features_b))
        #print("Intersection (" + app_a + " and " + app_b + "): " + str(intersec))
        return intersec
        #print(set(neighbor_features_a) & set(neighbor_features_b))

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
            return features + self.getNeighborFeaturesByLevel(graph, neighbors, k-1)

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
            #if 'https://schema.org/Review' not in neighbor :
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