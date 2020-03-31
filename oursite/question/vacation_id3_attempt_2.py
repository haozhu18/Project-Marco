import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.tree import export_graphviz
from sklearn.externals.six import StringIO
from scipy import sparse


'''
Generates a decision tree using the sklearn implementation. In order to 
train the decision tree, data is generated using normal perturbation of 
collected sample scores. The normal perturbation of the feature is 
distributed as N(0,Var(Feature)), that is to say, we assume the sample 
features match the true mean of the scores, the scores are normally distributed
and their variance is proportional to the sample variance. This assumption 
is accurate under certain assumptions, but we hope to have true training data 
in future implementations.
'''

#Dictionary renaming the possible categories.
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

#Reads in previously computed scores for each country.
scores = pd.read_csv('full_country_score.csv')

#Computes mean and standard deviation for each score category.
STD_DICT=dict(scores.std(axis=0,skipna=True))
MEAN_DICT=dict(scores.mean(axis=0,skipna=True))
RANGE_DICT={}

#Perturb scores to randomly generate training data.
for k,v in STD_DICT.items():
    RANGE_DICT[k]=(MEAN_DICT[k]-2*v, MEAN_DICT[k]-v, MEAN_DICT[k], \
                                            MEAN_DICT[k]+v,MEAN_DICT[k]+2*v)


for col in [col for col in scores.columns if col!='city']:
    scores[col]=(scores[col]-scores[col].min()) / \
                (scores[col].max()-scores[col].min())*10

all_scores = pd.DataFrame().append([scores]*200)
feature_cols = scores.columns[1:] 
scores[feature_cols] = np.sqrt(scores[feature_cols])
variance = pd.DataFrame([scores.var(axis = 0)])
for columns in feature_cols:
    all_scores[columns] += np.random.normal( \
                        0, np.sqrt(variance[columns] / 2), len(all_scores))

X = all_scores[feature_cols] 

y = all_scores['city']

X_train , X_test , y_train, y_test = train_test_split(X, y, test_size=0.3, \
                                                             random_state = 1)

clf = DecisionTreeClassifier(max_depth = 12, criterion='entropy', \
                                                        min_samples_leaf = 5)
clf = clf.fit(X_train,y_train)

y_pred = clf.predict(X_test)

# The actual tree structure.
tree_ = clf.tree_ 
# Child structure : leftchild[i] is the left child to node with absolute 
# path i. etc.
n_nodes = clf.tree_.node_count 
children_left = clf.tree_.children_left
children_right = clf.tree_.children_right
# Features: Feature[i] returns the column number/feature on which node i 
# is being split.
feature = clf.tree_.feature 
# Threshold: The value at which the split occurs.
threshold = clf.tree_.threshold 

# Gives names to feature columns rather than numbers.
feature_names = [feature_cols[i] for i in feature]#

#Identifies all the possible leaf nodes of the dataset.
leave_id = clf.apply(X_test) 


def add_noise(dictionary):
    '''
    Adds noise to sample data to generate traning data.
    '''

    for k,v in dictionary.items():
        dictionary[k]=dictionary[k]+np.random.normal(0,STD_DICT[k]/2)


def look_for_city2(node, dictionary):
    '''
    Helper function for look_for_city that only adds noise once.
    '''

    add_noise(dictionary)
    return look_for_city(node, dictionary)


def look_for_city(node, dictionary):
    '''
    Recursively over the outputted sklearn tree to find the correct node/class/
    city based on user preferences. Sklearn trees are rather odd in their
    implementation. Nodes are given absolute numerical values rather than
    some numerical structure, and one can find nodes and their children 
    through an array lookup structure, as seen above.
    '''

    if node in leave_id or children_left[node] == -1 or children_right[node] == -1:
        return (False, clf.classes_[np.argmax(tree_.value[node])])
    else:
        if feature_names[node] in dictionary:
            response=dictionary[feature_names[node]]
            if response <= threshold[node]:
                return look_for_city(node = children_left[node], \
                                                        dictionary=dictionary)
            else:
                return look_for_city(node = children_right[node], \
                                                        dictionary=dictionary)
        else:
            return(True, feature_names[node])
