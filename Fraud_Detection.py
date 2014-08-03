#import mtTkinter as Tkinter 

import pandas as pd
import seaborn
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import os
import time
import cPickle as pickle
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import cross_validation
from sklearn.metrics import roc_curve, auc
import d3py
from sqlalchemy import create_engine
#from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
#from matplotlib.figure import Figure as figure


STAT_THRESHOLD = 10 # This is the threshold to identify the anomalies from the statistical analysis
GRAPH_THRESHOLD = 2 # 2X the std out of PCA

def FindStatisticalAnomalies (df,approach='mean'):
	# This function is meant to find the statistical anomalies by various approaches
    if approach == 'mean':
        return df[df.values>df.values.mean()+3*df.values.std()].index
    elif approach == 'std':
        return df[df.values>(df.values.std())].index

def GetData(filename):
	# this function returns the dataset and removed any rows where there are NA's in the caller or callee column
	df = pd.read_csv(filename, converters={'calldno': str,'callgno': str} )
	df= df.dropna(subset = ["callgno","calldno"])
	df.callgno.apply(lambda x : x.encode('utf-8'))
	df.calldno.apply(lambda x : x.encode('utf-8'))

	print df.shape
	return df

def Statistical_Analysis(df):
	# Number of Callees
	x = df[["callgno","calldno"]].groupby("callgno")
	df_NumberOfUniqueCallees = x.apply(lambda x: len(set(x['calldno'])))
	df_NumberOfUniqueCallees = df_NumberOfUniqueCallees.order(ascending=False)
	temp = df_NumberOfUniqueCallees[0:100] # this is to plot just the top 100 phone numbers with the largest number of unique callees
#	print "after stat analysis"
	temp.to_csv("./data/Num_Of_Unique_Callees.csv")
	fig, ax = plt.subplots()
#	plt.figure()
#	temp.plot(kind='bar')
	rects1 = plt.bar(range(0,len(temp)), temp.values,color='r')
	ax.set_xticklabels(temp.index )
	plt.xlabel('Phone Numbers')
	plt.ylabel('Number of Unique Callees')
	plt.title('Number of Unique Callees per Phone Number')
	plt.savefig("./static/NumOfUniqueCallees.png",bbox_inches='tight')
	print "after stat analysis"
	plt.close()

	print "after stat analysis close"

	##### Second Alert is to look at the conversation duration for specific Called Numbers. The idea here is that fraudsters who leave missed calls or entice those to call them back, would have a higher cumulative call duration than the average.
	# Cumulative Conversation Duration

	x = df.groupby("calldno").sum()
	sorted_conv_duration = x.sort("cvrsn_dur",ascending=False)["cvrsn_dur"]
#	plt.figure()
#	sorted_conv_duration.plot(kind='bar')

	fig, ax = plt.subplots()
	rects1 = ax.bar(range(0,len(sorted_conv_duration[0:100])), sorted_conv_duration.values[0:100], log = False,color='r')
	ax.set_xticklabels( sorted_conv_duration.index[0:100] )
	plt.xlabel('Phone Numbers')
	plt.ylabel('Call Duration in Minutes')
	plt.title('Call Duration per Phone Number')
	plt.savefig("./static/CallDuration.png",bbox_inches='tight')
	plt.close()

	# Number of Unique Callers
	x = df[["callgno","calldno"]].groupby("calldno")
	df_NumberOfUniqueCallers = x.apply(lambda x: len(set(x['callgno'])))
	df_NumberOfUniqueCallers = df_NumberOfUniqueCallers.order(ascending=False)
	temp = df_NumberOfUniqueCallers[0:100]
	# plt.figure()
	# temp.plot(kind='bar')

	fig, ax = plt.subplots()
	rects1 = ax.bar(range(0,len(temp)), temp.values,color='r')
	ax.set_xticklabels(temp.index )
	plt.xlabel('Phone Numbers')
	plt.ylabel('Number of Unique Callers')
	plt.title('Number of Unique Callers per Phone Number')
	plt.savefig("./static/NumOfUniqueCallers.png",bbox_inches='tight')
	#plt.show()
	plt.close('all')

	print "after unique callers"

	# Find the Anomalies and return this as a list of anomalous phone numbers
	Anomalies = np.union1d(df_NumberOfUniqueCallees[0:STAT_THRESHOLD].index,sorted_conv_duration[0:STAT_THRESHOLD].index)
	Anomalies = np.union1d(Anomalies, df_NumberOfUniqueCallers[0:STAT_THRESHOLD].index)
	return Anomalies

def Generate_Edgelist (df,filename) :
	edgelist = df[["callgno","calldno"]]
	edgelist = edgelist.astype(str)
	edgelist.to_csv(filename,index=False)

# Now for the graph analysis

def Generate_Graph (total_edges_filename,Anomalies):
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


def Find_Graph_Anomalies (graph_metrics_df):
	# This function takes in the graph metrics and performs an anomaly analysis to identify the graph anomalies and returns those nodes
	# should scale the metrics first and then do one class SVM
	from sklearn.decomposition import PCA, KernelPCA
	X=graph_metrics_df.values[:,[1,2]]
	pca = PCA(n_components=2)
	pca.fit(X)
	X_lowed = pca.transform(np.real(X))

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


def FindMLAnomalies (df,feature_set, threshold, classifier_file):
	# the threshold defines how important we define the ML anomalies. Also note that only calling numbers are defined as Anomalies. Further work is needed to flag Called numbers
	Classifier = pickle.load( open( classifier_file, "rb" ) )
	df = DF_Preprocessing(df)
	X = df[feature_set].values
	probs = Classifier.predict_proba(X)	
	return df[probs[:,1]==threshold].callgno.unique()

def DF_Preprocessing (df_final):
	# Some data cleanup and binarization of categorical data
	ind = df_final[df_final['answind'] == 'N'].index
	df_final.loc[ind,'answind']=0
	ind1 = df_final[df_final['answind'] == 'Y'].index
	df_final.loc[ind1,'answind']=1
	df_final = df_final.fillna(0)
	return df_final

def TrainMLClassifier (df,Confirmed_Fraudster_Phone_Numbers,feature_set,classifier_file):
# This function labels the fraudster phone numbers and trains the classifer and pickles it
	df_new=df
	df_new['label'] = 0
	for num in Confirmed_Fraudster_Phone_Numbers:
		ind = df_new[df_new['callgno'] == num].index
		df_new.loc[ind,'label']=1
		ind2 = df_new[df_new['calldno'] == num].index
		df_new.loc[ind2,'label']=1
	print df_new.label.value_counts()
	y = df_new.pop('label')
	df_new = DF_Preprocessing(df[feature_set])
	X = np.array(df_new)
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

	print "Running Random Forest Classification ..."
	clf  = RandomForestClassifier(n_estimators=10,min_samples_leaf=3)

	clf = clf.fit(X_train, y_train)
	scores = cross_validation.cross_val_score(clf, X, y, cv=5)
	print "%s -- %s" % (clf.__class__, np.mean(scores))
	probas_ = clf.predict_proba(X_test)
	# Compute ROC curve and area the curve
	fpr, tpr, thresholds = roc_curve(y_test, probas_[:, 1])
	roc_auc = auc(fpr, tpr)
	print("Area under the ROC curve : %f" % roc_auc)
# 	fig, ax = plt.subplots()
# 	# Plot ROC curve
	plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
	plt.plot([0, 1], [0, 1], 'k--')
	plt.xlim([0.0, 1.0])
	plt.ylim([0.0, 1.0])
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title('Receiver operating characteristic example')
	plt.legend(loc="lower right")
#	plt.show()
	plt.savefig("./static/ROC.png",bbox_inches='tight')
	plt.close('all')
	pickle.dump(clf, open( "RF_phone_Fraud.pickle", "wb" ) )
	

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

def Database_Insertion (df,table):
	engine = create_engine('postgresql://skanajan:abcdef@localhost:5432/fraud_detector')
	df.to_sql(table,engine, if_exists='replace')
	print "in Database_Insertion"	

def RunComplete ():
	print "In Run Complete"

	start = time.time()
	df = GetData("hack_small.csv")[0:600]

	classifier_file = "RF_phone_Fraud.pickle"
	Classifier = pickle.load( open( classifier_file, "rb" ) )
	StatAnomalies = Statistical_Analysis(df)
	feature_set = ['answind','origpricingdestid','routingdestcd','supp_orgno','cvrsn_dur','attempts','cust_orgno','pricingdestid']

	MLAnomalies = FindMLAnomalies(df,feature_set,1,classifier_file)

	ML_Stat_Anomalies = set(StatAnomalies).union(set(MLAnomalies))

	print "Anomalies are " , StatAnomalies
	graph_filename = "edgelist.csv"
	Generate_Edgelist(df,graph_filename)
	Anomalies_Subgraph = Generate_Graph(graph_filename, ML_Stat_Anomalies)
	GraphAnalysisResults = GraphAnalysis(Anomalies_Subgraph)
	GraphAnomalies = Find_Graph_Anomalies(GraphAnalysisResults)
	Database_Insertion(pd.DataFrame(GraphAnomalies,columns=['possible_fraudster_phone_number']),'possible_fraudsters')

	print "graph anomalies are " , GraphAnomalies
	# Visualize these Graph Anomalies and let the user identify these anomalies
	#Draw_Suspect_Fraud_Node(GraphAnomalies[0],Anomalies_Subgraph)


	Confirmed_Fraudster_Phone_Numbers = GraphAnomalies # assumption for now

	TrainMLClassifier (df,Confirmed_Fraudster_Phone_Numbers,feature_set,classifier_file)


	print time.time() - start

#RunComplete()