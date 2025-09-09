import pandas as pd
   import numpy as np
   import matplotlib.pyplot as plt
   from sklearn.cluster import KMeans, AgglomerativeClustering
   from sklearn.preprocessing import StandardScaler
   from sklearn.metrics import silhouette_score

   data = pd.read_csv("/content/weight-height.csv")

   X = data[['Height', 'Weight']]

   scaler = StandardScaler()
   X_scaled = scaler.fit_transform(X)

   male_data = data[data['Gender'] == 'Male']
   female_data = data[data['Gender'] == 'Female']

   plt.figure(figsize=(12, 6))
   plt.hist(male_data['Height'], bins=30, alpha=0.5, label='Male')
   plt.hist(female_data['Height'], bins=30, alpha=0.5, label='Female')
   plt.legend()
   plt.title('Распределение веса по полу')
   plt.show()

   plt.figure(figsize=(12, 6))
   plt.hist(male_data['Weight'], bins=30, alpha=0.5, label='Male')
   plt.hist(female_data['Weight'], bins=30, alpha=0.5, label='Female')
   plt.legend()
   plt.title('Распределение роста по полу')
   plt.show()

   kmeans = KMeans(n_clusters=2, random_state=0)
   kmeans.fit(X_scaled)
   kmeans_silhouette = silhouette_score(X_scaled, kmeans.labels_)

   agg = AgglomerativeClustering(n_clusters=2)
   agg.fit(X_scaled)
   agg_silhouette = silhouette_score(X_scaled, agg.labels_)

   print(f"KMeans оценка: {kmeans_silhouette}")
   print(f"Agglomerative оценка: {agg_silhouette}")
