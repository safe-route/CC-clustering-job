# Machine Learning
from sklearn.cluster import DBSCAN

# Data library
import pandas as pd
import numpy as np

# Visualizing
import matplotlib.pyplot as plt

# Converting
import json

# Utility
import random

# DBScan parameters
EPSILON = 200 # meter
MIN_POINTS = 3 # minimum locations/points for core

# Unit conversion
ANGLE_TO_METER_RATIO = 0.00001 / 1.11 # source : https://www.usna.edu/Users/oceano/pguth/md_help/html/approx_equivalents.htm

# Files
dataset_file = "./dataset/crime_history.csv"
model_file = "./model/clustering.json"
statistic_file = "./model/area_statistic.json"

# Open dataset
data = pd.read_csv(dataset_file)

# Drop NaN value
data = data.dropna()

# Copy data before dropped
raw_data = data.copy()

# Drop unused columns
data = raw_data.drop(labels=['id', 'date', 'time', 'type', 'districts', 'subdistrict'], axis=1)

def meter_to_angle(meter):
    """Convert meter to angle"""
    return meter * ANGLE_TO_METER_RATIO

def angle_to_meter(angle):
    """Convvert angle to meter"""
    return angle / ANGLE_TO_METER_RATIO

def create_model() -> DBSCAN:
    """Creating DBSCAN model"""
    dbscan = DBSCAN(eps=meter_to_angle(EPSILON), min_samples=MIN_POINTS)
    dbscan = dbscan.fit(data)
    
    return dbscan

model = create_model()

def model_centroids(model):
    """Return the "centroids" of the model"""
    # Obtaining labels
    labels = model.labels_
    unique_labels = set(labels)
    # Generate colors
    colors = [tuple(plt.cm.Spectral(each)) for each in np.linspace(0, 1, len(unique_labels))]
    # Shuffle colors
    random.shuffle(colors)
    
    # Calculate centroids
    return_data = {'centroids': []}
    for label in unique_labels:
        if label == -1:
            # Skip noise
            continue
        # Calculate centroids coordinate and range
        label_points = data[labels==label]
        centroid = np.mean(label_points, axis=0)
        # Max distance from centroid to cluster member
        max_distance = np.sqrt(np.sum(np.square(label_points - centroid), axis=1)).to_numpy().flatten().max()
        avg_point = np.mean(label_points, axis=0)
        
        # Calculate crime info for each centroids
        crime_info = raw_data[labels==label].groupby('type').size().to_dict()
        
        # Generating data
        return_data['centroids'].append({
            'id': int(label),
            'latitude': float(avg_point['latitude']),
            'longitude': float(avg_point['longitude']),
            'range': float(angle_to_meter(max_distance)),
            'crime_info': crime_info
        })
    
    return return_data
    
centroids = model_centroids(model)

# Saving model using json
with open(model_file, 'w') as f:
    json.dump(centroids, f, indent=2)
