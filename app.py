import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns 

# Load the model from the file 
kmeans_model = joblib.load('bmx_kmean.joblib')

st.title('K-Means Clustering')
# Upload the dataset and save as csv 
uploaded_file = st.file_uploader('choose a CSV File', type='csv')

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)

    # Select The columnd fro clustering 
    clustering_columns =['bmxleg', 'bmxwaist']

    # Select the colums to be used for clustering
    # X = data[['bmxleg', 'bmxwaist']].dropna()
    data = data.dropna(subset=clustering_columns)

    # Check if tjere are enough data points fro clustering 
    if data.shape[0] > kmeans_model.n_clusters:
        # Perform clustering
        clustering_labels = kmeans_model.predict (data[clustering_columns])

    # Predict the cluster for each data point
   #clusters = kmeans_model.predict(X)

    # add cluster lables to the Dataframe
    data['cluster'] = clustering_labels

    st.write(data)

    # plot the clusters
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.write("Scaterplot of BMXLEG and BMXWAIST")
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=data, x='bmxleg', y='bmxwaist', hue='cluster')
    plt.title('Kmean Clustering')
    st.pyplot()
else:
    st.write("Not Enough data points fro clustering")