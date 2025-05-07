#!/usr/bin/env python3
"""
Spatial Clustering Script for Glacial Erratics

Provides functions to perform various clustering algorithms (DBSCAN, K-Means, Hierarchical)
on the erratic dataset based on geographic coordinates or other features.
"""

import sys
import os
import json
import typing
import argparse
import logging
import numpy as np
import pandas as pd
import geopandas as gpd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add parent directory for utils
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from python.utils.data_loader import load_erratics, json_to_file

# --- DBSCAN Clustering ---

def perform_dbscan_clustering(erratics_gdf: gpd.GeoDataFrame, eps: typing.Optional[float] = None, min_samples: int = 3) -> typing.Dict:
    """
    Performs DBSCAN clustering on erratic coordinates.
    Automatically estimates eps if not provided.

    Args:
        erratics_gdf: GeoDataFrame of erratics with geometry.
        eps: The maximum distance between samples for one to be considered as in the neighborhood of the other. 
             If None, it will be estimated using the k-distance graph.
        min_samples: The number of samples in a neighborhood for a point to be considered as a core point.

    Returns:
        Dictionary containing cluster assignments and metrics.
          {
              "algorithm": "DBSCAN",
              "params": {"eps": calculated_eps, "min_samples": min_samples},
              "num_clusters": count,
              "num_noise_points": count,
              "assignments": {erratic_id: cluster_label (-1 for noise), ...},
              "silhouette_score": score (optional, requires >1 cluster)
          }
    """
    logger.info(f"Performing DBSCAN clustering (min_samples={min_samples}, eps={eps or 'auto'})...")
    results = {
        "algorithm": "DBSCAN",
        "params": {"min_samples": min_samples},
        "assignments": {},
        "num_clusters": 0,
        "num_noise_points": 0,
        "silhouette_score": None
    }
    
    if erratics_gdf.empty or len(erratics_gdf) < min_samples:
        logger.warning("Not enough data points for DBSCAN.")
        results["error"] = "Insufficient data points"
        return results

    try:
        from sklearn.cluster import DBSCAN
        from sklearn.neighbors import NearestNeighbors
    except ImportError:
        logger.error("scikit-learn is required for DBSCAN clustering.")
        results["error"] = "scikit-learn not installed"
        return results

    # Use projected coordinates for distance calculation if possible, otherwise use haversine on lat/lon
    # For simplicity with DBSCAN's `metric` param, we'll use lat/lon with haversine for now.
    # A projected CRS would be better for Euclidean distance based eps.
    coords = np.radians(erratics_gdf[['geometry']].apply(lambda p: (p.geometry.y, p.geometry.x), axis=1).tolist()) # Lat, Lon in radians

    calculated_eps = eps
    if calculated_eps is None:
        # Estimate eps using k-distance graph (k=min_samples)
        k = min_samples * 2 # Factor used by NearestNeighbors often relates to min_samples
        try:
            logger.info(f"Estimating eps using k-distance graph (k={k})...")
            nn = NearestNeighbors(n_neighbors=k, metric='haversine')
            nn.fit(coords)
            distances, _ = nn.kneighbors(coords)
            # Get the distance to the k-th nearest neighbor for each point
            k_distances = np.sort(distances[:, -1]) 
            
            # Option 1: Use kneed library if available
            try:
                from kneed import KneeLocator
                knee = KneeLocator(range(len(k_distances)), k_distances, curve='convex', direction='increasing')
                if knee.knee:
                    calculated_eps = k_distances[knee.knee]
                    logger.info(f"Estimated eps (knee point): {calculated_eps:.5f} radians")
                else: # Fallback if knee not found
                     # Use a percentile (e.g., 95th) or median as fallback heuristic
                     calculated_eps = np.percentile(k_distances, 90)
                     logger.info(f"Knee point not found, using 90th percentile eps: {calculated_eps:.5f} radians")

            except ImportError:
                 # Option 2: Fallback heuristic if kneed not available (e.g., median or percentile)
                 calculated_eps = np.percentile(k_distances, 90) 
                 logger.warning("kneed library not found for optimal eps estimation. Using 90th percentile heuristic.")
                 logger.info(f"Estimated eps (90th percentile): {calculated_eps:.5f} radians")

        except Exception as e:
            logger.error(f"Error estimating eps: {e}. Using default heuristic value.")
            # Provide a very rough default based on typical coordinate ranges if estimation fails
            calculated_eps = 0.01 # Radians, adjust based on expected data density

    results["params"]["eps"] = calculated_eps
    
    # Apply DBSCAN
    try:
        db = DBSCAN(eps=calculated_eps, min_samples=min_samples, metric='haversine')
        cluster_labels = db.fit_predict(coords)
        
        results["assignments"] = {int(erratics_gdf.iloc[i]['id']): int(label) for i, label in enumerate(cluster_labels)}
        
        # Calculate metrics
        unique_labels = set(cluster_labels)
        results["num_clusters"] = len(unique_labels) - (1 if -1 in unique_labels else 0)
        results["num_noise_points"] = int(np.sum(cluster_labels == -1))
        
        logger.info(f"DBSCAN Result: Found {results['num_clusters']} clusters and {results['num_noise_points']} noise points.")

        # Calculate silhouette score if more than 1 cluster and fewer than n-1 points are noise
        n_samples = len(coords)
        if results["num_clusters"] > 1 and results["num_clusters"] < (n_samples - 1):
             # Need to exclude noise points for silhouette calculation if metric allows
             # sklearn's silhouette_score handles labels directly
             try:
                 score = silhouette_score(coords, cluster_labels, metric='haversine')
                 results["silhouette_score"] = float(score)
                 logger.info(f"Silhouette Score: {score:.3f}")
             except ValueError as sil_err:
                  logger.warning(f"Could not calculate silhouette score: {sil_err}")
             except Exception as e:
                  logger.error(f"Unexpected error calculating silhouette score: {e}")
        elif results["num_clusters"] <= 1:
             logger.info("Silhouette score requires more than 1 cluster.")

    except Exception as e:
        logger.error(f"Error during DBSCAN execution: {e}")
        results["error"] = f"DBSCAN execution failed: {e}"

    return results

# --- K-Means Clustering ---

def perform_kmeans_clustering(erratics_gdf: gpd.GeoDataFrame, n_clusters: int = 5, features: typing.List[str] = ['longitude', 'latitude']) -> typing.Dict:
    """
    Performs K-Means clustering on specified features of the erratics.

    Args:
        erratics_gdf: GeoDataFrame of erratics. Must contain columns specified in `features`.
        n_clusters: The number of clusters to form.
        features: List of column names to use for clustering (e.g., ['longitude', 'latitude', 'elevation']).

    Returns:
        Dictionary containing cluster assignments and metrics.
          {
              "algorithm": "KMeans",
              "params": {"n_clusters": n_clusters, "features": features},
              "assignments": {erratic_id: cluster_label, ...},
              "silhouette_score": score,
              "cluster_centers": [[feat1_center, feat2_center,...], ...] 
          }
    """
    logger.info(f"Performing K-Means clustering (k={n_clusters}) using features: {features}...")
    results = {
        "algorithm": "KMeans",
        "params": {"n_clusters": n_clusters, "features": features},
        "assignments": {},
        "silhouette_score": None,
        "cluster_centers": []
    }

    if erratics_gdf.empty or len(erratics_gdf) < n_clusters:
        logger.warning("Not enough data points for K-Means.")
        results["error"] = "Insufficient data points"
        return results

    try:
        from sklearn.cluster import KMeans
    except ImportError:
        logger.error("scikit-learn is required for K-Means clustering.")
        results["error"] = "scikit-learn not installed"
        return results

    # Prepare data
    try:
        if 'longitude' in features and 'latitude' in features and 'geometry' in erratics_gdf.columns:
            # Use geometry coordinates directly if requested
            data_for_clustering = erratics_gdf[['geometry']].apply(lambda p: [p.geometry.x, p.geometry.y], axis=1, result_type='expand')
            data_for_clustering.columns = ['longitude', 'latitude']
             # Combine with other features if requested
            other_features = [f for f in features if f not in ['longitude', 'latitude']]
            if other_features:
                 data_for_clustering = pd.concat([data_for_clustering, erratics_gdf[other_features]], axis=1)
            # Ensure final feature list matches request order if important
            data_for_clustering = data_for_clustering[features] 
        else:
             data_for_clustering = erratics_gdf[features].copy()

        data_for_clustering.dropna(inplace=True) # Drop rows with NaNs in selected features
        ids_for_clustering = erratics_gdf.loc[data_for_clustering.index, 'id']

        if len(data_for_clustering) < n_clusters:
             logger.warning(f"Not enough data points ({len(data_for_clustering)}) after handling NaNs for K-Means (k={n_clusters}).")
             results["error"] = "Insufficient data points after NaN removal"
             return results

        # Scale features
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(data_for_clustering)
    except KeyError as e:
        logger.error(f"Feature column '{e}' not found in erratic data.")
        results["error"] = f"Missing feature column: {e}"
        return results
    except Exception as e:
         logger.error(f"Error preparing data for K-Means: {e}")
         results["error"] = f"Data preparation failed: {e}"
         return results

    # Apply K-Means
    try:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10) # n_init='auto' in newer sklearn
        cluster_labels = kmeans.fit_predict(scaled_data)
        
        results["assignments"] = {int(ids_for_clustering.iloc[i]): int(label) for i, label in enumerate(cluster_labels)}
        results["cluster_centers"] = scaler.inverse_transform(kmeans.cluster_centers_).tolist() # Store original scale centers
        
        # Calculate silhouette score
        if n_clusters > 1 and len(scaled_data) > n_clusters:
             try:
                 score = silhouette_score(scaled_data, cluster_labels)
                 results["silhouette_score"] = float(score)
                 logger.info(f"Silhouette Score: {score:.3f}")
             except Exception as e:
                  logger.error(f"Error calculating silhouette score: {e}")
        
        logger.info(f"K-Means Result: Assigned {len(results['assignments'])} points to {n_clusters} clusters.")

    except Exception as e:
        logger.error(f"Error during K-Means execution: {e}")
        results["error"] = f"K-Means execution failed: {e}"

    return results


# --- Hierarchical Clustering ---

def perform_hierarchical_clustering(erratics_gdf: gpd.GeoDataFrame, n_clusters: int = 5, linkage: str = 'ward', features: typing.List[str] = ['longitude', 'latitude']) -> typing.Dict:
    """
    Performs Agglomerative Hierarchical clustering.

    Args:
        erratics_gdf: GeoDataFrame of erratics.
        n_clusters: The number of clusters to find.
        linkage: Linkage criterion ('ward', 'complete', 'average', 'single'). 'ward' minimizes variance within clusters.
        features: List of column names to use for clustering.

    Returns:
        Dictionary containing cluster assignments and metrics.
          {
              "algorithm": "Hierarchical",
              "params": {"n_clusters": n_clusters, "linkage": linkage, "features": features},
              "assignments": {erratic_id: cluster_label, ...},
              "silhouette_score": score
          }
    """
    logger.info(f"Performing Hierarchical clustering (k={n_clusters}, linkage={linkage}) using features: {features}...")
    results = {
        "algorithm": "Hierarchical",
        "params": {"n_clusters": n_clusters, "linkage": linkage, "features": features},
        "assignments": {},
        "silhouette_score": None
    }

    if erratics_gdf.empty or len(erratics_gdf) < n_clusters:
        logger.warning("Not enough data points for Hierarchical Clustering.")
        results["error"] = "Insufficient data points"
        return results

    try:
        from sklearn.cluster import AgglomerativeClustering
    except ImportError:
        logger.error("scikit-learn is required for Hierarchical clustering.")
        results["error"] = "scikit-learn not installed"
        return results
        
    # Prepare data (similar to K-Means)
    try:
        if 'longitude' in features and 'latitude' in features and 'geometry' in erratics_gdf.columns:
            data_for_clustering = erratics_gdf[['geometry']].apply(lambda p: [p.geometry.x, p.geometry.y], axis=1, result_type='expand')
            data_for_clustering.columns = ['longitude', 'latitude']
            other_features = [f for f in features if f not in ['longitude', 'latitude']]
            if other_features:
                 data_for_clustering = pd.concat([data_for_clustering, erratics_gdf[other_features]], axis=1)
            data_for_clustering = data_for_clustering[features]
        else:
             data_for_clustering = erratics_gdf[features].copy()

        data_for_clustering.dropna(inplace=True)
        ids_for_clustering = erratics_gdf.loc[data_for_clustering.index, 'id']

        if len(data_for_clustering) < n_clusters:
             logger.warning(f"Not enough data points ({len(data_for_clustering)}) after handling NaNs for Hierarchical Clustering (k={n_clusters}).")
             results["error"] = "Insufficient data points after NaN removal"
             return results

        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(data_for_clustering)
    except KeyError as e:
        logger.error(f"Feature column '{e}' not found in erratic data.")
        results["error"] = f"Missing feature column: {e}"
        return results
    except Exception as e:
         logger.error(f"Error preparing data for Hierarchical Clustering: {e}")
         results["error"] = f"Data preparation failed: {e}"
         return results

    # Apply Agglomerative Clustering
    try:
        # For Ward linkage, affinity must be 'euclidean'
        # Other linkages can use different affinities/distance metrics if needed
        affinity = 'euclidean'
        if linkage != 'ward' and 'latitude' in features: # Use haversine for geographic coords if not Ward
             affinity = 'haversine' 
             # Caution: AgglomerativeClustering with haversine might be slow
             # Use scaled_data? Haversine expects radians lat/lon, not scaled arbitrary features.
             # Revert to euclidean for simplicity if features != [lon, lat]
             if features != ['longitude', 'latitude']:
                 logger.warning(f"Using euclidean distance for linkage '{linkage}' with mixed features. Consider feature engineering.")
                 affinity = 'euclidean'
             else:
                 # Use radians for haversine
                 coords = np.radians(data_for_clustering[['latitude', 'longitude']].values) 
                 scaled_data = coords # Use radians directly for haversine
        
        model = AgglomerativeClustering(n_clusters=n_clusters, affinity=affinity, linkage=linkage)
        cluster_labels = model.fit_predict(scaled_data)
        
        results["assignments"] = {int(ids_for_clustering.iloc[i]): int(label) for i, label in enumerate(cluster_labels)}
        
        # Calculate silhouette score
        if n_clusters > 1 and len(scaled_data) > n_clusters:
             try:
                 score = silhouette_score(scaled_data, cluster_labels, metric=affinity if affinity != 'precomputed' else 'euclidean')
                 results["silhouette_score"] = float(score)
                 logger.info(f"Silhouette Score: {score:.3f}")
             except Exception as e:
                  logger.error(f"Error calculating silhouette score: {e}")

        logger.info(f"Hierarchical Clustering Result: Assigned {len(results['assignments'])} points to {n_clusters} clusters.")

    except Exception as e:
        logger.error(f"Error during Hierarchical execution: {e}")
        results["error"] = f"Hierarchical execution failed: {e}"
        
    return results

# --- Main Execution ---

def main():
    parser = argparse.ArgumentParser(description='Perform clustering on glacial erratics.')
    parser.add_argument('--algorithm', type=str, required=True, choices=['dbscan', 'kmeans', 'hierarchical'], help='Clustering algorithm to use.')
    parser.add_argument('--output', type=str, required=True, help='Output file for results (JSON)')
    
    # DBSCAN args
    parser.add_argument('--eps', type=float, help='DBSCAN: Epsilon parameter (distance). Auto-estimated if not provided.')
    parser.add_argument('--min_samples', type=int, default=3, help='DBSCAN: Minimum number of samples.')
    
    # K-Means / Hierarchical args
    parser.add_argument('--k', type=int, default=5, help='K-Means/Hierarchical: Number of clusters.')
    parser.add_argument('--features', nargs='+', default=['longitude', 'latitude'], help='K-Means/Hierarchical: Features to use for clustering.')
    parser.add_argument('--linkage', type=str, default='ward', choices=['ward', 'complete', 'average', 'single'], help='Hierarchical: Linkage criterion.')

    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        
    logger.info(f"Loading all erratic data for clustering...")
    erratics_gdf = load_erratics()
    
    if erratics_gdf.empty:
        logger.error("No erratics data loaded. Cannot perform clustering.")
        results = {"error": "No erratic data available"}
    elif args.algorithm == 'dbscan':
        results = perform_dbscan_clustering(erratics_gdf, eps=args.eps, min_samples=args.min_samples)
    elif args.algorithm == 'kmeans':
        results = perform_kmeans_clustering(erratics_gdf, n_clusters=args.k, features=args.features)
    elif args.algorithm == 'hierarchical':
        results = perform_hierarchical_clustering(erratics_gdf, n_clusters=args.k, linkage=args.linkage, features=args.features)
    else:
        results = {"error": f"Unknown algorithm: {args.algorithm}"}

    logger.info(f"Writing clustering results to {args.output}")
    json_to_file(results, args.output)
    
    if 'error' in results:
        logger.error(f"Clustering failed: {results['error']}")
        print(json.dumps(results, indent=2)) # Also print error to stdout
        return 1
        
    logger.info("Clustering analysis complete.")
    print(json.dumps(results, indent=2)) # Print success results to stdout
    return 0

if __name__ == "__main__":
    sys.exit(main()) 