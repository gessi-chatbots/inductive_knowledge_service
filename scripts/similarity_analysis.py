import json, requests, sys, os
from telnetlib import DO
import matplotlib.pyplot as plt

####
# USAGE: python similarity_analysis.py <k> <LEVEL> <documentType> <folder>
# PARAMETERS
KG_INDUCTIVE_KNOWLEDGE_SERVICE = 'http://localhost:5001/api/computeSimilarity'
GRAPHDB_REPOSITORY  = 'http://localhost:8080/getTopKSimilarApps'
LEVEL = int(sys.argv[2])
K = int(sys.argv[1])
DOCUMENT = sys.argv[3]
prefix = 'https://schema.org/MobileApplication/'
# END PARAMETERS

data = json.load(open('representative_apps.json', 'r', encoding='utf-8'))

kg_inductive_knowledge_response = requests.post(KG_INDUCTIVE_KNOWLEDGE_SERVICE, params={'level':LEVEL, 'k':K}, json = list(data.keys())).json()
graphdb_repository_response = requests.get(GRAPHDB_REPOSITORY, params={'k':K, 'documentType':DOCUMENT}, json = list(data.keys())).json()

def get_recall_rate_k(data, response):
    recall_rate_k = {}
    for app in data.keys():
        category = data[app]
        recall_rate = []
        match = 0
        
        for similar_app in response[prefix + app]:
            if category in similar_app['category']:
                match += 1
            recall_rate.append(match)
        recall_rate_k[app] = recall_rate
    return recall_rate_k

recall_rate_k_kgservice = get_recall_rate_k(data, kg_inductive_knowledge_response)
recall_rate_k_graphdb = get_recall_rate_k(data, graphdb_repository_response)

#with open('recall-rate-k-kgservice.json', 'w') as outfile:
#            json.dump(recall_rate_k_kgservice, outfile)

#with open('recall-rate-k-graphdb.json', 'w') as outfile:
#            json.dump(recall_rate_k_graphdb, outfile)

x = list(range(1,K+1))
path = sys.argv[4] + "_" + DOCUMENT

for app in recall_rate_k_graphdb.keys():
    graphdb = recall_rate_k_graphdb[app]
    if len(graphdb) != K:
        graphdb = [0] * K
    kgservice = recall_rate_k_kgservice[app]
    
    plt.figure()
    plt.plot(x, graphdb, label = "GraphDB")
    plt.plot(x, kgservice, label = "NetworkX")
    plt.plot(x, x, linestyle="dashed")
    plt.xlabel("K")
    plt.ylabel("#apps from same category")
    plt.title("recall-rate@k for " + app)
    plt.legend()
    plt.gca().set_ylim([0,K])

    if not os.path.exists(path):
        os.makedirs(path)
    plt.savefig(path + os.sep + "recall_rate_k-" + app + ".png")