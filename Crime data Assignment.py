import pandas as pd
import matplotlib.pylab as plt

data = pd.read_csv(r"C:\Users\kaval\OneDrive\Desktop\360digit\datatypes\crime_data.csv")

data.describe()  
data.info()



# Normalization function 
def norm_func(i):
    x = (i-i.min())	/ (i.max()-i.min())
    return (x)

# Normalized data frame (considering the numerical part of data)
df_norm = norm_func(data.iloc[:, 1:])
df_norm.describe()

# for creating dendrogram 
from scipy.cluster.hierarchy import linkage
import scipy.cluster.hierarchy as sch 

z = linkage(df_norm, method = "complete", metric = "euclidean")

# Dendrogram
plt.figure(figsize=(15, 8));plt.title('Hierarchical Clustering Dendrogram');plt.xlabel('Index');plt.ylabel('Distance')
sch.dendrogram(z, 
    leaf_rotation = 0,  # rotates the x axis labels
    leaf_font_size = 10 # font size for the x axis labels
)
plt.show()


# Now applying AgglomerativeClustering choosing 5 as clusters from the above dendrogram
from sklearn.cluster import AgglomerativeClustering

h_complete = AgglomerativeClustering(n_clusters = 3, linkage = 'complete', affinity = "euclidean").fit(df_norm) 
h_complete.labels_

cluster_labels = pd.Series(h_complete.labels_)

data['clust'] = cluster_labels # creating a new column and assigning it to new column 

df = data.iloc[:, [5,0,1,2,3,4]]
df.head()

# Aggregate mean of each cluster
df.iloc[:, 2:].groupby(data.clust).mean()



import os
os.getcwd()
