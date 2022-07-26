import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


telco = pd.read_excel(r"C:\Users\kaval\OneDrive\Desktop\360digit\datatypes\Telco_customer_churn.xlsx")

telco.describe()


# drop customerid  column
telco.drop(['Customer ID','Count','Quarter'], axis=1, inplace=True)
telco.dtypes

#@create dummies
tele_new=pd.get_dummies(telco)
tele_new1=pd.get_dummies(telco , drop_first=True)


#Normalization function 
def norm_func(i):
    x = (i-i.min())	/ (i.max()-i.min())
    return (x)
            
# Normalized data frame (considering the numerical part of data)
tele_norm = norm_func(tele_new1.iloc[:, :])
tele_norm.describe()

#for creating dendograms
from scipy.cluster.hierarchy import linkage
import scipy.cluster.hierarchy as sch 

z = linkage(tele_norm, method = "complete", metric = "euclidean")

# Dendrogram
plt.figure(figsize=(15, 8));plt.title('Hierarchical Clustering Dendrogram');plt.xlabel('Index');plt.ylabel('telephone')
sch.dendrogram(z, 
    leaf_rotation = 0,  # rotates the x axis labels
    leaf_font_size = 10 # font size for the x axis labels
)
plt.show()

# Now applying AgglomerativeClustering choosing 5 as clusters from the above dendrogram
from sklearn.cluster import AgglomerativeClustering

h_complete = AgglomerativeClustering(n_clusters = 5, linkage = 'complete', affinity = "euclidean").fit(tele_norm) 

h_complete.labels_

cluster_labels = pd.Series(h_complete.labels_)

telco['clust'] = cluster_labels # creating a new columnz



telco.columns.values
telco=telco[['clust','Referred a Friend', 'Number of Referrals', 'Tenure in Months',
       'Offer', 'Phone Service', 'Avg Monthly Long Distance Charges',
       'Multiple Lines', 'Internet Service', 'Internet Type',
       'Avg Monthly GB Download', 'Online Security', 'Online Backup',
       'Device Protection Plan', 'Premium Tech Support', 'Streaming TV',
       'Streaming Movies', 'Streaming Music', 'Unlimited Data',
       'Contract', 'Paperless Billing', 'Payment Method',
       'Monthly Charge', 'Total Charges', 'Total Refunds',
       'Total Extra Data Charges', 'Total Long Distance Charges',
       'Total Revenue']]
# Aggregate mean of each cluster
telco.iloc[:, 2:].groupby(telco.clust).mean()

# creating a csv file 
tele_new1.to_csv("tele.csv")

import os
os.getcwd()
    