

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

import Get_Data as get_data
import statistical_analysis
import graph_analysis
import ml_analysis
import logging

'''
This is the main function loop that runs the entire data flow from beginning to end. Refer to the data flow diagram in the readme to refer to the various components.
'''


def Database_Insertion (df,table):
	"""
	Description: This function inserts the possible possible_fraudsters identified from the graph analysis into the database.
	INPUT: 
	       df - type: DataFrame - the list of fraudster ID's
	       table  - type: string - the table to be inserted into
	OUTPUT: No output. Just stores the data in the database
	"""
	engine = create_engine('postgresql://skanajan:abcdef@localhost:5432/fraud_detector')
	df.to_sql(table,engine, if_exists='replace')
	logging.debug("in Database_Insertion")

def RunComplete ():

	"""
	Description: This is the main function that ties all the flows together. Currently the flows are tied together by passing around lists of anomalies or fraudsters. Longer term
	work would entail in making this as database queries.
	INPUT: 
	       df - type: DataFrame - the list of fraudster ID's
	       table  - type: string - the table to be inserted into
	OUTPUT: No output. Just stores the data in the database
	"""
	logging.debug ("In Run Complete")

	start = time.time()
	df = get_data.GetData("hack_small.csv")[0:400000]

	classifier_file = "RF_phone_Fraud.pickle"
	Classifier = pickle.load( open( classifier_file, "rb" ) )
	StatAnomalies = statistical_analysis.Statistical_Analysis(df)
	feature_set = ['answind','origpricingdestid','routingdestcd','supp_orgno','cvrsn_dur','attempts','cust_orgno','pricingdestid']

	MLAnomalies = ml_analysis.FindMLAnomalies(df,feature_set,1,classifier_file)

	ML_Stat_Anomalies = set(StatAnomalies).union(set(MLAnomalies))
	print "Total Anomalies are ", len(ML_Stat_Anomalies)
	logging.debug ("Anomalies are " , StatAnomalies)
	graph_filename = "edgelist.csv"
	graph_analysis.Generate_Edgelist(df,graph_filename)
	Anomalies_Subgraph = graph_analysis.Generate_Graph(graph_filename, ML_Stat_Anomalies)
	GraphAnalysisResults = graph_analysis.GraphAnalysis(Anomalies_Subgraph)
	GraphAnomalies = graph_analysis.Find_Graph_Anomalies(GraphAnalysisResults)
	Database_Insertion(pd.DataFrame(GraphAnomalies,columns=['possible_fraudster_phone_number']),'possible_fraudsters')

	logging.debug ("graph anomalies are " , GraphAnomalies)
	# Visualize these Graph Anomalies and let the user identify these anomalies
	#Draw_Suspect_Fraud_Node(GraphAnomalies[0],Anomalies_Subgraph)


	Confirmed_Fraudster_Phone_Numbers = GraphAnomalies # assumption for now

	ml_analysis.TrainMLClassifier (df,Confirmed_Fraudster_Phone_Numbers,feature_set,classifier_file)


	logging.debug (time.time() - start)



RunComplete()
