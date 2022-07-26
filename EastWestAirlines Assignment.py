import pandas as pd
import matplotlib.pylab as plt

Univ1 = pd.read_excel(r"C:\Users\kaval\OneDrive\Desktop\360digit\datatypes\EastWestAirlines.xlsx", sheet_name = 'data')

#  "C:\Users\kaval\OneDrive\Desktop\360digit\datatypes\University_Clustering.xlsx" 

Univ1.describe()
Univ1.info()

# Univ = Univ1.drop(["State"], axis=1)

# Normalization function 
def norm_func(i):
    x = (i-i.min())	/ (i.max()-i.min())
    return (x)

# Normalized data frame (considering the numerical part of data)
df_norm = norm_func(Univ1.iloc[:, :])
df_norm.describe()

# for creating dendrogram 
from scipy.cluster.hierarchy import linkage
import scipy.cluster.hierarchy as sch 

z = linkage(df_norm, method = "complete", metric = "euclidean")

# Dendrogram
plt.figure(figsize=(15, 12));plt.title('Hierarchical Clustering Dendrogram');plt.xlabel('Index');plt.ylabel('Distance')
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

Univ1['clust'] = cluster_labels # creating a new column and assigning it to new column 

Univ2 = Univ1.iloc[:, [12,0,1,2,3,4,5,6,7,8,9,10,11]]
Univ2.head()

# Aggregate mean of each cluster
Univ2.iloc[:, 2:].groupby(Univ1.clust).mean()

# creating a csv file 
Univ2.to_csv("EastWestAirlines.csv", encoding = "utf-8")

import os
os.getcwd()
