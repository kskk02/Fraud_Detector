''' This file defines the anomaly detection using graph metrics '''
import pandas as pd
import seaborn
import networkx as nx
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import time
from sqlalchemy import create_engine
import Get_Data as get_data
import statistical_analysis
import graph_analysis
import numpy as np
#import pdb

GRAPH_THRESHOLD = 2 # 2X the std out of PCA


def Generate_Edgelist (df,filename) :
	"""
	Description: This function exports an edgelist from the DataFrame.
	INPUT: 
	       df - type: DataFrame - This is the cleaned up datafram of the call log that contains the caller, callee columns
	       filename  - type: string - This is the filename that will store the edgelist
	OUTPUT: No output. Just stores the file in the file system
	"""

	edgelist = df[["callgno","calldno"]]
	edgelist = edgelist.astype(str)
	edgelist.to_csv(filename,index=False)

# Now for the graph analysis

def Generate_Graph (total_edges_filename,Anomalies):
	"""
	Description: This function generates a subgraph of all the anomaly nodes.
	INPUT: 
			total_edges_filename: This is a filename that has the edges. Comes from Generate_Edgelist
			Anomalies: This is a list of nodes that are anomalies
	OUTPUT: returns the subgraph in networkx format
	"""
	# this function builds the subgraph for the anomalies set. TBD : Build the subgraph using a longer time window
	TotalGraph = nx.read_edgelist(total_edges_filename,delimiter=',')
	Anomalies_Subgraph = nx.Graph()
	for anomaly in Anomalies:
		Anomalies_Subgraph = nx.compose(Anomalies_Subgraph,nx.ego_graph(TotalGraph,anomaly,radius=3))
	print len(Anomalies_Subgraph.nodes())
	return Anomalies_Subgraph

def GraphAnalysis(Anomalies_Subgraph):
	# Generate an output where the indexes are the Anomalies phone numbers and the columns are the graph metrics
	pagerank_matrix = nx.pagerank(Anomalies_Subgraph,max_iter=200)
	pagerank_matrix_df = pd.DataFrame.from_dict(pagerank_matrix,orient='index')
	pagerank_matrix_df.column=["Page Rank"]

	trianglecount_matrix = nx.triangles(Anomalies_Subgraph)
	trianglecount_matrix_df = pd.DataFrame.from_dict(trianglecount_matrix,orient='index')
	trianglecount_matrix_df.column=["Triangle Count"]
	trianglecount_matrix_df.describe()

	body = pd.merge(pagerank_matrix_df,trianglecount_matrix_df,right_index=True,left_index=True)
	body.head()
	body = body.reset_index()
	body.columns = ["Phone Number", "Page Rank", "Triangle Count"]
	return body

def Draw_Suspect_Fraud_Node (suspect, Anomalies_Subgraph):
	import d3py
	import networkx as nx   
	import random 
	Anomalies_Subgraph = nx.ego_graph(Anomalies_Subgraph,suspect,radius=3)
	name = str(random.random()) + "_graph"
	with d3py.NetworkXFigure(Anomalies_Subgraph, name=name,width=1000, height=1000) as p:
	    p += d3py.ForceLayout()
	    p.css['.node'] = {'fill': 'blue', 'stroke': 'magenta'}
	    p.show() 


def Find_Graph_Anomalies (graph_metrics_df):
	# This function takes in the graph metrics and performs an anomaly analysis to identify the graph anomalies and returns those nodes
	# should scale the metrics first and then do one class SVM
	from sklearn.decomposition import PCA, KernelPCA
	X=graph_metrics_df.values[:,[1,2]]
	pca = PCA(n_components=2)
	pca.fit(X)

	X_lowed = pca.transform(np.real(X))
	#pdb.set_trace()
	temp = X_lowed.copy()
	temp[:,1]=temp[:,0]
	temp[:,0]=range(0,len(temp))
	np.savetxt("./static/PCA_Results.csv", temp, delimiter=",", header = "Index, PCA Transform")
#	fig, ax = plt.subplots()
#	plt.plot(pca.explained_variance_)
	plt.plot(X_lowed[:,0],'o')
	plt.xlabel('Index')
	plt.ylabel('Highest Explanatory PCA Dimension')
	plt.title('Anomalies on Graph Metrics using PCA')
	plt.savefig("./static/PCA_Analysis.png",bbox_inches='tight')
	plt.close('all')

	NumOfFrauds = len(X_lowed[abs(X_lowed[:,0]) > (GRAPH_THRESHOLD*abs(np.std(X_lowed[:,0])))])
	print "num of frauds ", NumOfFrauds
	indices = np.argsort(X_lowed[:,0])[:len(X_lowed)-(NumOfFrauds+1):-1]
	graph_metrics_df["Phone Number"].apply(lambda x : x.encode('utf-8'))
	SuspectFraudPhoneNumbers = graph_metrics_df.iloc[indices]["Phone Number"].values
	return SuspectFraudPhoneNumbers
