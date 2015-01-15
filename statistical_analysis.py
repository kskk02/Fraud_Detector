
''' This component performs the statistical analysis and generates a list of anomalous phone numbers per a statistical treatment.
In this case the statistical anomaly detection approach is to identify phone numbers that have an above mean level of callers, callees
and cumulative call duration. '''	

import matplotlib
import pandas as pd
import seaborn
import matplotlib.pyplot as plt
import numpy as np
import os
import time



STAT_THRESHOLD = 10 # This is the threshold to identify the anomalies from the statistical analysis

def FindStatisticalAnomalies (df,approach='mean'):
	''' This function defines the mechanism of defining the anomalies. E.g. 3X the mean of more than the standard deviation. 
	'''
	# This function is meant to find the statistical anomalies by various approaches
	if approach == 'mean':
		return df[df.values>df.values.mean()+3*df.values.std()].index
	elif approach == 'std':
		return df[df.values>(df.values.std())].index


def Statistical_Analysis(df):
	''' This is the core function that performs the statistical analysis. The current approach is to take the data set and 
	derive outliers by looking at the number of unique callers, unique callees and cumulative duration for every unique phone number. 
	THis function also generates the plots and saves the figures. 

	Input: Processed Dataframe from the call log.
	'''
	# Number of Callees
	x = df[["callgno","calldno"]].groupby("callgno")
	df_NumberOfUniqueCallees = x.apply(lambda x: len(set(x['calldno'])))
	df_NumberOfUniqueCallees = df_NumberOfUniqueCallees.order(ascending=False)

	temp = df_NumberOfUniqueCallees[0:100] # this is to plot just the top 100 phone numbers with the largest number of unique callees
#	print "after stat analysis"
	temp.to_csv("./static/Num_Of_Unique_Callees.csv", header=["Num of Callees"])
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
	print "dasdas das" , sorted_conv_duration.head()
	sorted_conv_duration[0:100].to_csv("./static/call_duration.csv", header=["Cumulative Call Duration"])
#	plt.figure()
#	sorted_conv_duration.plot(kind='bar')
	# call_dur_df = pd.DataFrame(sorted_conv_duration.index,sorted_conv_duration.values)
	# call_dur_df.to_csv("./data/call_duration.csv", header=["Cumulative Call Duration"])

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
	temp.to_csv("./static/Num_Of_Unique_Callers.csv", header=["Num of Callers"])
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
