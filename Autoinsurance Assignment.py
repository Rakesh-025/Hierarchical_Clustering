import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df=pd.read_csv(r"C:\Users\kaval\OneDrive\Desktop\360digit\datatypes\AutoInsurance.csv")
df.describe()
df.dtypes

#drop columns 
df.drop(["Customer","State",'Effective To Date'],axis=1, inplace=True)

df.dtypes

#@create dummies
df_new=pd.get_dummies(df)
df_new1=pd.get_dummies(df , drop_first=True)

#Normalization function 
def norm_func(i):
    x = (i-i.min())	/ (i.max()-i.min())
    return (x) 

# Normalized data frame (considering the numerical part of data)
df_norm = norm_func(df_new1.iloc[:, :])
df_norm.describe()

#for creating dendograms
from scipy.cluster.hierarchy import linkage
import scipy.cluster.hierarchy as sch 

z = z = linkage(df_norm, method = "complete", metric = "euclidean")

# Dendrogram
plt.figure(figsize=(15, 8));plt.title('Hierarchical Clustering Dendrogram');plt.xlabel('Index');plt.ylabel('auto insurance')
sch.dendrogram(z, 
    leaf_rotation = 0,  # rotates the x axis labels
    leaf_font_size = 10 # font size for the x axis labels
)
plt.show()

# Now applying AgglomerativeClustering choosing  10 as clusters from the above dendrogram
from sklearn.cluster import AgglomerativeClustering

h_complete = AgglomerativeClustering(n_clusters = 10, linkage = 'complete', affinity = "euclidean").fit(df_norm) 
h_complete.labels_

cluster_labels = pd.Series(h_complete.labels_)

df['clust'] = cluster_labels # creating a new column and assigning it to new column 

df.columns.values
df=df['clust','Customer Lifetime Value', 'Response', 'Coverage', 'Education',
       'EmploymentStatus', 'Gender', 'Income', 'Location Code',
       'Marital Status', 'Monthly Premium Auto',
       'Months Since Last Claim', 'Months Since Policy Inception',
       'Number of Open Complaints', 'Number of Policies', 'Policy Type',
       'Policy', 'Renew Offer Type', 'Sales Channel',
       'Total Claim Amount', 'Vehicle Class', 'Vehicle Size']

# Aggregate mean of each cluster
df.iloc[:, 2:].groupby(df.clust).mean()
# creating a csv file 
df_new1.to_csv("autoinsurance")

import os
os.getcwd
