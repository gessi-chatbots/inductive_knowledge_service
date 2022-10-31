import json, requests, sys, os
from telnetlib import DO
import matplotlib.pyplot as plt
import numpy as np

def get_recall_rate_k(data, response, prefix, k=20):
    recall_rate_k = {}
    aggregated = [0]*k
    for app in data.keys():
        category = data[app]
        recall_rate = []
        match = 0

        #workaround for features
        if 'DefinedTerm' in prefix:
            app = app.replace(' ', '')

        for similar_app in response[prefix + app]:
            if category in similar_app['category']:
                match += 1
            recall_rate.append(match)
        recall_rate_k[app] = recall_rate

        # compute aggregate
        for i in range(0, len(recall_rate_k[app])):
            aggregated[i] += recall_rate_k[app][i]

    recall_rate_k['AGGREGATED'] = aggregated
    return recall_rate_k

####
# USAGE: python similarity_analysis.py <k> <LEVEL>
# PARAMETERS
KG_INDUCTIVE_KNOWLEDGE_SERVICE_APPS_BY_FEATURE = 'http://localhost:5001/api/getTopKAppsByFeature'
KG_INDUCTIVE_KNOWLEDGE_SERVICE_SIMILAR_APPS = 'http://localhost:5001/api/getTopKSimilarApps'
GRAPHDB_REPOSITORY_APPS_BY_FEATURE  = 'http://localhost:8080/findAppsByFeature'
GRAPHDB_REPOSITORY_SIMILAR_APPS  = 'http://localhost:8080/findSimilarApps'
LEVEL = int(sys.argv[2])
K = int(sys.argv[1])
NETWORK_FILE_W_REVIEWS = 'networkx-data-with-reviews.json'
NETWORK_FILE_WO_REVIEWS = 'networkx-data-without-reviews.json'
PREFIX_APP = 'https://schema.org/MobileApplication/'
PREFIX_FEATURE = 'https://schema.org/DefinedTerm/'
# END PARAMETERS

def experiment_round(data='representative_apps.json', url='http://localhost:7200/repositories/app-data-repository-without-reviews', 
network_file=NETWORK_FILE_WO_REVIEWS, path='results/similar_apps_without_reviews/', graph_db_endpoint = GRAPHDB_REPOSITORY_SIMILAR_APPS, 
kg_endpoint = KG_INDUCTIVE_KNOWLEDGE_SERVICE_SIMILAR_APPS, prefix = PREFIX_APP, k = 20):
    data = json.load(open(data, 'r', encoding='utf-8'))
    requests.post(kg_endpoint, params={'url':url})
    kg_inductive_knowledge_response = requests.post(kg_endpoint, params={'level':LEVEL, 'k':k, 'network-file': network_file, 'algorithm':'simrank'}, json = list(data.keys())).json()
    graphdb_description_repository_response = requests.get(graph_db_endpoint, params={'k':k, 'documentType':'DESCRIPTION'}, json = list(data.keys())).json()
    graphdb_summary_repository_response = requests.get(graph_db_endpoint, params={'k':k, 'documentType':'SUMMARY'}, json = list(data.keys())).json()
    graphdb_changelog_repository_response = requests.get(graph_db_endpoint, params={'k':k, 'documentType':'CHANGELOG'}, json = list(data.keys())).json()
    graphdb_all_repository_response = requests.get(graph_db_endpoint, params={'k':k, 'documentType':'ALL'}, json = list(data.keys())).json()

    recall_rate_k_kgservice = get_recall_rate_k(data, kg_inductive_knowledge_response, prefix, k)
    recall_rate_k_graphdb_description = get_recall_rate_k(data, graphdb_description_repository_response, prefix, k)
    recall_rate_k_graphdb_summary = get_recall_rate_k(data, graphdb_summary_repository_response, prefix, k)
    recall_rate_k_graphdb_changelog = get_recall_rate_k(data, graphdb_changelog_repository_response, prefix, k)
    recall_rate_k_graphdb_all = get_recall_rate_k(data, graphdb_all_repository_response, prefix, k)

    if not os.path.exists(path):
            os.makedirs(path)

    with open(path + os.sep + 'recall-rate-k-kgservice.json', 'w') as outfile:
        json.dump(recall_rate_k_kgservice, outfile)
    with open(path + os.sep +  'recall-rate-k-graphdb_description.json', 'w') as outfile:
        json.dump(recall_rate_k_graphdb_description, outfile)
    with open(path +  os.sep + 'recall-rate-k-graphdb_summary.json', 'w') as outfile:
        json.dump(recall_rate_k_graphdb_summary, outfile)
    with open(path +  os.sep + 'recall-rate-k-graphdb_changelog.json', 'w') as outfile:
        json.dump(recall_rate_k_graphdb_changelog, outfile)
    with open(path +  os.sep + 'recall-rate-k-graphdb_all.json', 'w') as outfile:
        json.dump(recall_rate_k_graphdb_all, outfile)

    x = list(range(1,k+1))

    for app in recall_rate_k_graphdb_summary.keys():
        graphdb_summary = recall_rate_k_graphdb_summary[app]
        graphdb_description= recall_rate_k_graphdb_description[app]
        graphdb_changelog = recall_rate_k_graphdb_changelog[app]
        graphdb_all = recall_rate_k_graphdb_all[app]

        if len(graphdb_changelog) != k:
            graphdb_changelog = [0] * k
        kgservice = recall_rate_k_kgservice[app]
        
        plt.figure()
        plt.plot(x, graphdb_description, label = "Index-Based (Description)")
        plt.plot(x, graphdb_summary, label = "Index-Based (SUMMARY)")
        plt.plot(x, graphdb_changelog, label = "Index-Based (CHANGELOG)")
        plt.plot(x, graphdb_all, label = "Index-Based (ALL)")
        plt.plot(x, kgservice, label = "Graph-Based")

        plt.xticks(np.arange(0, k+1, k/10))
        if app == 'AGGREGATED':
            y = []
            for element in x:
                y.append(element * (len(recall_rate_k_graphdb_summary)-1))
            plt.plot(x, y, linestyle="dashed")
            plt.yticks(np.arange(0, k*len(recall_rate_k_graphdb_summary), k*(len(recall_rate_k_graphdb_summary)-1)/10))
            plt.gca().set_ylim([0,k*(len(recall_rate_k_graphdb_summary)-1)])
        else:
            plt.plot(x, x, linestyle="dashed")
            plt.yticks(np.arange(0, k+1, k/10))
            plt.gca().set_ylim([0,k])
        plt.xlabel("K")
        plt.ylabel("#apps from same category")
        plt.title("recall-rate@k for " + app)

        plt.legend()
        plt.savefig(path + os.sep + "recall_rate_k-" + app + ".png")
        plt.close()

#################################################
###### EXPERIMENT 1: Apps by Feature without reviews
#################################################
print("Begin EXPERIMENT 1: Apps by Feature without reviews")
experiment_round(data='representative_features.json', 
url='http://localhost:7200/repositories/app-data-repository-without-reviews', 
network_file=NETWORK_FILE_WO_REVIEWS, 
path='results/without_reviews/apps_by_feature/', 
graph_db_endpoint=GRAPHDB_REPOSITORY_APPS_BY_FEATURE, 
kg_endpoint=KG_INDUCTIVE_KNOWLEDGE_SERVICE_APPS_BY_FEATURE,
prefix = PREFIX_FEATURE,
k = K)
print("Finished!")

#################################################
###### EXPERIMENT 2: Similar Apps without reviews
#################################################
print("Begin EXPERIMENT 2: Similar Apps without reviews")
experiment_round(data='representative_apps.json', 
url='http://localhost:7200/repositories/app-data-repository-without-reviews', 
network_file=NETWORK_FILE_WO_REVIEWS, 
path='results/without_reviews/similar_apps/', 
graph_db_endpoint=GRAPHDB_REPOSITORY_SIMILAR_APPS, 
kg_endpoint=KG_INDUCTIVE_KNOWLEDGE_SERVICE_SIMILAR_APPS,
prefix = PREFIX_APP,
k = K)
print("Finished!")

#################################################
###### EXPERIMENT 3: Apps by Feature with reviews
#################################################
print("Begin EXPERIMENT 3: Apps by Feature withreviews")
experiment_round(data='representative_features.json', 
url='http://localhost:7200/repositories/app-data-repository-with-reviews', 
network_file=NETWORK_FILE_W_REVIEWS, 
path='results/with_reviews/apps_by_feature/', 
graph_db_endpoint=GRAPHDB_REPOSITORY_APPS_BY_FEATURE, 
kg_endpoint=KG_INDUCTIVE_KNOWLEDGE_SERVICE_APPS_BY_FEATURE,
prefix = PREFIX_FEATURE,
k = K)
print("Finished!")

#################################################
###### EXPERIMENT 4: Similar Apps with reviews
#################################################
print("Begin EXPERIMENT 4: Similar Apps with reviews")
experiment_round(data='representative_apps.json', 
url='http://localhost:7200/repositories/app-data-repository-with-reviews', 
network_file=NETWORK_FILE_W_REVIEWS, 
path='results/with_reviews/similar_apps/', 
graph_db_endpoint=GRAPHDB_REPOSITORY_SIMILAR_APPS, 
kg_endpoint=KG_INDUCTIVE_KNOWLEDGE_SERVICE_SIMILAR_APPS,
prefix = PREFIX_APP,
k = K)
print("Finished!")