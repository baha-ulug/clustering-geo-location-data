'''
ignore warnings
'''
import warnings
warnings.filterwarnings("ignore")

'''
read geoloc and haverdist data file from same git repo
file type: json
'''

# imports
import git
import json
import os
import matplotlib.pyplot as plt

# define repo url and local path for saving data
repo_url = "../geolocation-clustering-v01.git"
local_path = "..Clustering based on Geolocation data\data" 
# local path should be an empty folder or sholud not exist and the command itself will create it

# clone repo to local path
git.Repo.clone_from(repo_url,local_path)

# open, read, and display file from cloned repo
f = open(os.path.join(local_path,"GeoLocData.json"))
geoloc = json.load(f)
f = open(os.path.join(local_path,"HaverDist.json"))
haver_dist = json.load(f)

# convert json file to pandas dataframe
import pandas as pd

geoloc_df = pd.DataFrame(data=geoloc)
columns = geoloc_df.columns.tolist()
long_lat_columns = columns[:2]

haver_dist_df = pd.DataFrame(data=haver_dist)
haver_dist_df['dist'] = haver_dist_df['distance'].map(lambda x: float(x[:-2]))
haver_dist_df.drop('distance', inplace=True, axis=1)

# scale geoloc data with sklearn min-max sacler to [0,1] range
from sklearn import preprocessing

scaler = preprocessing.MinMaxScaler()
scaler.fit(geoloc_df[long_lat_columns])
scaled = scaler.transform(geoloc_df[long_lat_columns])
scaled_df = pd.DataFrame(data=scaled, columns=long_lat_columns)
scaled_df['parentId'] = geoloc_df['parentId']

'''
# plot scaled geolocs

plt.scatter(x=scaled_df[long_lat_columns[0]], y=scaled_df[long_lat_columns[1]])
plt.xlabel(long_lat_columns[0])
plt.ylabel(long_lat_columns[1])
plt.show()
'''

'''
find max distance between parents
inputs:
       parent_id_list: list of parent ids to find every pair distance
'''

def max_dist(parent_id_list):
  max_d = 0
  for i,id in enumerate(parent_id_list):
    #new_list = parent_id_list.iloc[i:].drop(index=i)
    new_list = parent_id_list.iloc[i+1:]
    distances = haver_dist_df.loc[(haver_dist_df['from'] == id) & (haver_dist_df['to'].isin(new_list))]
    if distances['dist'].max()>max_d:
      max_d = distances['dist'].max()
  return max_d
  
'''  
cluster entered df based on entered column names and number of clusters
'''
from sklearn.cluster import KMeans

def clustering(df,column_names,n_clusters):
  kmeans = KMeans(n_clusters = n_clusters, init ='k-means++', random_state = 0)
  kmeans.fit(df[column_names]) # Compute k-means clustering.
  df['cluster_label'] = kmeans.fit_predict(df[column_names])
  return df
  
'''  
define parameters
'''
MAX_DISTANCE = 5 # max distance between one cluster members (5KM)
MAX_GROUP_SIZE = 4 # max number of cluster members (4 people)

'''
clustering based on max group size and max distance in each group
'''
scaled_df = clustering(df=scaled_df,column_names=long_lat_columns,n_clusters=2)
grouped = scaled_df.groupby(scaled_df.columns[-1])
grouped = [g[1] for g in grouped]
final_groups_list = []

for g in grouped:
  new_df = g.reset_index(drop=True)
  # if len of new_df is less than or equal to MAX_GROUP_SIZE and max distance in new_df is less than MAX_DISTANCE then create the group
  if len(new_df) <= MAX_GROUP_SIZE and max_dist(new_df['parentId'])<=MAX_DISTANCE:
    final_groups_list.append(new_df['parentId'].tolist())
  # otherwise Recluster
  else:
    new_clusters = clustering(df=new_df,column_names=long_lat_columns,n_clusters=2)
    new_grouped = new_clusters.groupby(new_clusters.columns[-1])
    new_grouped = [g[1] for g in new_grouped]
    grouped += new_grouped
#print(f'final_groups_list:  {final_groups_list}')

'''
create a dataframe from created groups and plot groups
'''

print(f'number of clusters: {len(final_groups_list)}')
final_groups_df = pd.DataFrame(columns=scaled_df.columns)
for i,group in enumerate(final_groups_list):
  new_df = scaled_df.loc[scaled_df['parentId'].isin(group)]
  new_df['cluster_label'] = i
  final_groups_df = pd.concat([final_groups_df,new_df])
final_groups_df = final_groups_df.reset_index(drop=True)

plt.rcParams['figure.figsize'] = [12, 6]
final_groups_df.plot.scatter(x = 'parentLongitude', y = 'parentLatitude', c=final_groups_df['cluster_label'], s=100, cmap='tab20')
plt.show()
