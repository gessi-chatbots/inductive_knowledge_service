import networkx as nx

class SimilarityService:
    
    ##  ALGORITHMS this class implements
    #   - Jaccard
    #   - Overlap / Szymkiewicz-Simpson coefficient
    #   - K-nearest neighbors
    #   - 

    nxGraph = nx.DiGraph()

    def addNodes(nodes):
        nxGraph.add_nodes_from(nodes)

    def addEdges(nodes):
        nxGraph.add_edges_from(nodes)


    def computeSimilarity(self):
        # TODO
        print("TODO")