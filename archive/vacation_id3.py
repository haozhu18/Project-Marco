import pandas as pd
import time
import numpy as np
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.tree import export_graphviz
from sklearn.externals.six import StringIO  
from IPython.display import Image  
from scipy import sparse
import pydotplus

RENAME_DIC={
        'NATURE_PARKS': 'Parks',
        'TOURS': 'Organized tours',
        'COLD_OUTDOOR': 'Cold weather',
        'SIGHTS_AND_LANDMARKS': 'Sightseeing',
        'AMUSEMENT_PARKS': 'Entertainment',
        'SHOPPING': 'Shopping',
        'LAND_OUTDOOR': 'The outdoors',
        'ZOOS':  'Animals',
        'GROUND_NATURE': 'Nature',
        'CASINOS': 'Nightlife',
        'OUTDOOR_ACTIVITIES': 'Sports and recreation',
        'HISTORIC': 'History',
        'SEA_NATURE': 'Oceans and marine life',
        'SEA_OUTDOOR': 'The beach',
        'CONCERTS_SHOWS': 'Music and theater',
        'FOOD_DRINK': 'Restaurants and bars',
        'MUSEUMS': 'Culture and museums'
        }

scores = pd.read_csv('full_country_score.csv')

STD_DICT=dict(scores.std(axis=0, skipna=True))
MEAN_DICT=dict(scores.mean(axis=0,skipna=True))
RANGE_DICT={}

for k,v in STD_DICT.items():
    RANGE_DICT[k]=(MEAN_DICT[k]-2*v,MEAN_DICT[k]-v,MEAN_DICT[k],MEAN_DICT[k]+v,MEAN_DICT[k]+2*v)

def get_score(column):
    i=int(input('1-5 rating of '+str(RENAME_DIC[column])+': '))
    return RANGE_DICT[column][i-1]



for col in [col for col in scores.columns if col!='city']:
    scores[col]=(scores[col]-scores[col].min())/(scores[col].max()-scores[col].min())*10

all_scores = pd.DataFrame().append([scores]*200)
feature_cols = scores.columns[1:] 
scores[feature_cols] = np.sqrt(scores[feature_cols])
variance = pd.DataFrame([scores.var(axis = 0)])
for columns in feature_cols:
    all_scores[columns] += np.random.normal(0,np.sqrt(variance[columns]/2),len(all_scores))


X = all_scores[feature_cols] 

y = all_scores['city']


X_train , X_test , y_train, y_test = train_test_split(X,y,test_size=0.3,random_state = 1)

clf = DecisionTreeClassifier(max_depth = 12,criterion='entropy',min_samples_leaf = 5)
clf = clf.fit(X_train,y_train)


y_pred = clf.predict(X_test)

tree_ = clf.tree_ # The actual tree structure.
n_nodes = clf.tree_.node_count # Child structure : leftchild[i] is the left child to node with absolute path i. etc.
children_left = clf.tree_.children_left
children_right = clf.tree_.children_right
feature = clf.tree_.feature # Features: Feature[i] returns the column number/feature on which node i is being split.
threshold = clf.tree_.threshold # Threshold: The value at which the split occurs.

feature_names = [feature_cols[i] for i in feature]# Gives names to feature columns rather than numbers.

leave_id = clf.apply(X_test) #Identifies all the possible leaf nodes of the dataset.

def traverse (k):
    '''
    Decision tree traversal. Sklearn uses absolute paths to store nodes. For example 
    the numbers 0,1,2, ... , number of nodes -1 are all specific nodes. It's really 
    stupid, but all the code below is based on that. Node 0 is the root, and
    everything else isn't easy to identify.
    '''

    city_output=set([])
    asked_dic={}

    def add_noise():
        for k,v in asked_dic.items():
            asked_dic[k]=asked_dic[k]+np.random.normal(MEAN_DICT[k],STD_DICT[k]/2)

    def recurse(node):
        
        if node in leave_id: #Base case: Hitting a leaf
            return clf.classes_[np.argmax(tree_.value[node])]
        else:#User input.
            if feature_names[node] in asked_dic:
                response=asked_dic[feature_names[node]]
            else:
                response=get_score(feature_names[node])
                asked_dic[feature_names[node]]=response
            if response <= threshold[node]:
                return recurse(node = children_left[node])
            else:
                return recurse(node = children_right[node])

    counter=0
    while len(city_output)<k and counter<3*k:
        add_noise()
        city_output.add(recurse(0))
        counter=counter+1
    return city_output
        

    # First let's retrieve the decision path of each sample. The decision_path
    # method allows to retrieve the node indicator functions. A non zero element of
    # indicator matrix at the position (i, j) indicates that the sample i goes
    # through the node j.

    #node_indicator = estimator.decision_path(X_test)

    # Similarly, we can also have the leaves ids reached by each sample.
