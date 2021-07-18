import os
import sys

import azureml as aml
from azureml.core import Workspace, Datastore, Dataset
from azureml.core.model import Model
from azureml.core.run import Run
import argparse
import json
import time
#import traceback
import logging

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, accuracy_score
import pandas as pd
import numpy as np
import re
import math
import seaborn as sn
import matplotlib.pyplot as plt
import joblib

"""
Titanic classification
"""

__author__="Benya Wang"
__email__="benya.wang@hotmail.com"

class TitanicClassification():
    def __init__(self,args):
        """[initialize steps]
        Args:
        1. initalize azure ml run object 
        2. create directories 
        """
        self.args=args 
        self.run=Run.get_context()
        self.workspace=self.run.experiment.workspace
        os.makedirs('./model_metas',exist_ok=True)
    
    def get_files_from_datastore(self,container_name,file_name):
        """[get input csv file from default datastore]

        Args:
            container_name ([type]): [description]
            file_name ([type]): [description]
        return data_ds:azure ml dataset object
        """
        datastore_paths=[(self.datastore, os.path.join(container_name,file_name))]
        data_ds=Dataset.Tabular.from_delimited_files(path=datastore_paths)
        dataset_name=self.args.dataset_name
        if dataset_name not in self.workspace.datasets:
            data_ds=data_ds.register(workspace=self.workspace,
                    name=dataset_name,
                    description=self.args.dataset_desc,
                    tags={'format':'csv'},
                    create_new_version=True)
        else:
            print('dataset {} already in workspace'.format(dataset_name))
        return data_ds

    def create_pipeline(self):
        # Titanic data training and validation 
        self.datastore=Datastore.get(self.workspace,self.workspace.get_default_datastore().name)
        print('received datastore')
        input_ds=self.get_files_from_datastore(self.args.container_name,self.args.input_csv)
        final_df=input_ds.to_pandas_dataframe()
        print('input df info',final_df.info())
        print('input df head',final_df.head())

        X=final_df[["Pclass","Age","SibSp","Parch","Fare"]]
        y=final_df[['Survived']]

        X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.4,random_state=1991)

        model=DecisionTreeClassifier()
        model.fit(X_train,y_train)
        y_pred=model.predict(X_test)
        print('model score:',model.score(X_test,y_test))
        joblib.dump(model,self.args.model_path)

        self.validate(y_test,y_pred,X_test)

        match = re.search('([^\/]*)$', self.args.model_path)
        self.run.upload_file(name=self.args.artifact_loc+match.group(1),
                            path_or_stream=self.args.model_path)
        
        print('run files',self.run.get_file_names())

        self.run.complete()

    def create_confusion_matrix(self, y_true, y_pred, name):
        '''
        Create confusion matrix
        '''
        try:
            confm = confusion_matrix(y_true, y_pred, labels=np.unique(y_pred))
            print("Shape : ", confm.shape)

            df_cm = pd.DataFrame(confm, columns=np.unique(y_true), index=np.unique(y_true))
            df_cm.index.name = 'Actual'
            df_cm.columns.name = 'Predicted'
            df_cm.to_csv(name+".csv", index=False)
            self.run.upload_file(name="./outputs/"+name+".csv",path_or_stream=name+".csv")

            plt.figure(figsize = (120,120))
            sn.set(font_scale=1.4)
            c_plot = sn.heatmap(df_cm, fmt="d", linewidths=.2, linecolor='black',cmap="Oranges", annot=True,annot_kws={"size": 16})
            plt.savefig("./outputs/"+name+".png")    
            self.run.log_image(name=name, plot=plt)
        except Exception as e:
            #traceback.print_exc()    
            logging.error("Create consufion matrix Exception")

    def create_outputs(self, y_true, y_pred, X_test, name):
        '''
        Create prediction results as a CSV
        '''
        pred_output = {"Actual Survived" : y_true['Survived'].values, "Predicted Survived": y_pred['Survived'].values}        
        pred_df = pd.DataFrame(pred_output)
        pred_df = pred_df.reset_index()
        X_test = X_test.reset_index()
        final_df = pd.concat([X_test, pred_df], axis=1)
        final_df.to_csv(name+".csv", index=False)
        self.run.upload_file(name="./outputs/"+name+".csv",path_or_stream=name+".csv")

    def validate(self, y_true, y_pred, X_test):
        self.run.log(name="Precision", value=round(precision_score(y_true, y_pred, average='weighted'), 2))
        self.run.log(name="Recall", value=round(recall_score(y_true, y_pred, average='weighted'), 2))
        self.run.log(name="Accuracy", value=round(accuracy_score(y_true, y_pred), 2))

        self.create_confusion_matrix(y_true, y_pred, "confusion_matrix")

        y_pred_df = pd.DataFrame(y_pred, columns = ['Survived'])
        self.create_outputs(y_true, y_pred_df,X_test, "predictions")
        self.run.tag("TitanicClassificationFinish") 

if __name__=='__main__':
    parser=argparse.ArgumentParser(description='QA Code Indexing pipeline')
    parser.add_argument('--container_name',type=str,help='path to default datastore contaienr')
    parser.add_argument('--input_csv',type=str,help='input csv file')
    parser.add_argument('--dataset_name',type=str,help='dataset name to store in wp')
    parser.add_argument('--ataset_desc',type=str,help="dataset description")
    parser.add_argument('--model_path',type=str,help='path to store the model')
    parser.add_argument('--artifact_loc',type=str,help='devops artifact location to store the model',
                        default='')
    args=parser.parse_args()

    titanic_classifer=TitanicClassification(args)
    titanic_classifer.create_pipeline()