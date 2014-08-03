
import d3py
import networkx as nx
import pandas as pd
import logging
import json
from flask import Flask
from flask import render_template,request
from sqlalchemy import create_engine
import Fraud_Detection
from pandas.io import sql


app = Flask(__name__)

global max_fraudsters_in_DB
max_fraudsters_in_DB=1
Fraud_Detection.RunComplete()
TotalGraph = nx.read_edgelist("edgelist.csv",delimiter=',')
print "initial max_fraudsters_in_DB is " ,max_fraudsters_in_DB
@app.route('/',methods=['GET','POST'])


def Main_Page():

    print "initial max_fraudsters_in_DB is " ,max_fraudsters_in_DB
    
    possible_fraudsters_index=0
    print "general"
    if request.method == 'POST':
        if request.form['submit'] == 'Confirm Fraudster Anomaly':
            possible_fraudsters_index = int(request.form['fraud_index']) + 1
            if possible_fraudsters_index >= max_fraudsters_in_DB:
                possible_fraudsters_index = 0
        elif request.form['submit'] == "Not Anomaly and Skip to Next":
            possible_fraudsters_index = int(request.form['fraud_index']) + 1
            if possible_fraudsters_index >= max_fraudsters_in_DB:
                print "possible_fraudsters_index is ", possible_fraudsters_index
                print "max_fraudsters_in_DB is ", max_fraudsters_in_DB
                possible_fraudsters_index = 0
            print "index is ", possible_fraudsters_index    
            pass
        elif request.form['submit'] == "Confirm Fraudster Anomaly":
            possible_fraudsters_index = int(request.form['fraud_index']) + 1
            if possible_fraudsters_index >= max_fraudsters_in_DB:
                print "possible_fraudsters_index is ", possible_fraudsters_index
                print "max_fraudsters_in_DB is ", max_fraudsters_in_DB
                possible_fraudsters_index = 0

            pass
        elif request.form['submit'] == 'Run Complete Detection':
            Fraud_Detection.RunComplete()
            possible_fraudsters_index =0
        else:
            pass # unknown


    def Generate_Graph (anomaly):
        # this function builds the subgraph for the anomalies set. TBD : Build the subgraph using a longer time window
        Anomalies_Subgraph = nx.ego_graph(TotalGraph,anomaly,radius=1)
        print "length of subgraph for " ,anomaly , " is " ,len(Anomalies_Subgraph.nodes())
        myjson = d3py.json_graph.node_link_data(Anomalies_Subgraph)
        return myjson        

    def _decode_list(data):
        rv = []
        for item in data:
            if isinstance(item, unicode):
                item = item.encode('utf-8')
            elif isinstance(item, list):
                item = _decode_list(item)
            elif isinstance(item, dict):
                item = _decode_dict(item)
            rv.append(item)
        return rv

    def _decode_dict(data):
        rv = {}
        for key, value in data.iteritems():
            if isinstance(key, unicode):
                key = key.encode('utf-8')
            if isinstance(value, unicode):
                value = value.encode('utf-8')
            elif isinstance(value, list):
                value = _decode_list(value)
            elif isinstance(value, dict):
                value = _decode_dict(value)
            rv[key] = value
        return rv

    def Find_Next_Fraudster (table,possible_fraudsters_index):
        global max_fraudsters_in_DB
        engine = create_engine('postgresql://skanajan:abcdef@localhost:5432/fraud_detector')
        cnx = engine.raw_connection()
        possible_fraudsters_df = sql.read_sql("SELECT * FROM " + table , cnx)
        print "max_fraudsters_in_DB before" , max_fraudsters_in_DB

        max_fraudsters_in_DB = sql.read_sql("SELECT count(*) FROM " + table , cnx)['count'][0]
        print "max_fraudsters_in_DB after" , max_fraudsters_in_DB
        cnx.close()
        print "fraudster index is " , possible_fraudsters_index
        fraudster_id = possible_fraudsters_df.iloc[possible_fraudsters_index,1].encode('utf-8')
        return fraudster_id


    Fraudster_ID = Find_Next_Fraudster ('possible_fraudsters',possible_fraudsters_index)
    print " New Fraudster_ID is ", Fraudster_ID 
    json_data = Generate_Graph(Fraudster_ID)

    # with open("network_data.json") as json_file:
    #     json_data = json.load(json_file,object_hook=_decode_dict)
    json_data = _decode_dict(json_data)
#    myjson = d3py.json_graph.node_link_data(G)
    myjson = json_data
    newdict = dict()
    newdict['links']=  myjson['links']
    newdict['nodes']= myjson['nodes']
#    newdict = myjson

#    with open('figure.json', 'w') as outfile:
#        json.dump(newdict, outfile)
    return render_template('force.html', data=newdict, Fraudster_index = possible_fraudsters_index, Fraudster_ID= Fraudster_ID )


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=6969, debug=True)