from http import client
from sklearn.cluster import DBSCAN
from google.cloud import storage

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import json
import random
import os

EPSILON = 200
MIN_POINTS = 3

ANGLE_TO_METER_RATIO = 0.00001 / 1.11

# Unit conversion
def meter_to_angle(meter):
    """Convert meter to angle"""
    return meter * ANGLE_TO_METER_RATIO

def angle_to_meter(angle):
    """Convvert angle to meter"""
    return angle / ANGLE_TO_METER_RATIO

# Model
def create_model(data) -> DBSCAN:
    """Creating DBSCAN model"""
    dbscan = DBSCAN(eps=meter_to_angle(EPSILON), min_samples=MIN_POINTS)
    dbscan = dbscan.fit(data)
    
    return dbscan

def model_centroids(model, data, raw_data):
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

def main(data):

    # Init storage client and read csv in memory
    client = storage.Client()
    bucket = client.get_bucket('gs://safe_route')
    dataset_file_read = pd.read_csv('gs://safe_route/crime_history.csv', encoding='utf-8')
    # statistic_uri
    model_file = 'gs://safe_route/JSON File/clustering.json'# clustering_uri

    data = dataset_file_read
    data = data.dropna()

    raw_data = data.copy()

    data = raw_data.drop(labels=['id', 'date', 'time', 'type', 'districts', 'subdistrict'], axis=1)

    model = create_model(data)

    centroids = model_centroids(model, data, raw_data)

     # Saving model using json
    with open(model_file, 'w') as f:
        json.dump(centroids, f, indent=2)

