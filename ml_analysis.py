
import pandas as pd
import seaborn
import networkx as nx
import matplotlib
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



def FindMLAnomalies (df,feature_set, threshold, classifier_file):
	# the threshold defines how important we define the ML anomalies. Also note that only calling numbers are defined as Anomalies. Further work is needed to flag Called numbers
	"""
	Description: This function exports an edgelist from the DataFrame.
	INPUT: 
	       df - type: DataFrame - This is the cleaned up datafram of the call log that contains the caller, callee columns
	       filename  - type: string - This is the filename that will store the edgelist
	OUTPUT: No output. Just stores the file in the file system
	"""


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
	
