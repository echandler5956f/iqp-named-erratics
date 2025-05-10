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

def perform_dbscan_clustering(erratics_gdf: gpd.GeoDataFrame, eps: typing.Optional[float] = None, 
                              min_samples: int = 3, features: typing.Optional[typing.List[str]] = None,
                              metric: str = 'auto') -> typing.Dict:
    """
    Performs DBSCAN clustering on erratic coordinates or other features.
    Automatically estimates eps if not provided.

    Args:
        erratics_gdf: GeoDataFrame of erratics with geometry.
        eps: The maximum distance between samples for one to be considered as in the neighborhood of the other. 
             If None, it will be estimated using the k-distance graph.
        min_samples: The number of samples in a neighborhood for a point to be considered as a core point.
        features: List of column names to use for clustering. If None, uses geographic coordinates.
                 Special handling for 'vector_embedding' which is expected to be a list/array column.
        metric: Distance metric to use. 'auto' chooses based on features:
               - 'haversine' for geographic coordinates (default when features=None)
               - 'euclidean' for scalar features
               - 'cosine' for vector embeddings

    Returns:
        Dictionary containing cluster assignments and metrics.
          {
              "algorithm": "DBSCAN",
              "params": {"eps": calculated_eps, "min_samples": min_samples, 
                         "features": features, "metric": metric},
              "num_clusters": count,
              "num_noise_points": count,
              "assignments": {erratic_id: cluster_label (-1 for noise), ...},
              "silhouette_score": score (optional, requires >1 cluster)
          }
    """
    logger.info(f"Performing DBSCAN clustering (min_samples={min_samples}, eps={eps or 'auto'})...")
    results = {
        "algorithm": "DBSCAN",
        "params": {
            "min_samples": min_samples,
            "features": features,
            "metric": metric
        },
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
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        logger.error("scikit-learn is required for DBSCAN clustering.")
        results["error"] = "scikit-learn not installed"
        return results

    # Determine what features to use and prepare data accordingly
    using_geographic_coords = False
    using_vector_embedding = False
    data_for_clustering = None
    effective_metric = metric

    # Default to geographic coordinates if no features specified
    if features is None or (len(features) == 2 and 
                            ('longitude' in features or 'lat' in features or 
                             'latitude' in features or 'lon' in features)):
        # Use geographic coordinates with Haversine distance
        using_geographic_coords = True
        logger.info("Using geographic coordinates for clustering.")
        coords = np.radians(erratics_gdf.geometry.apply(lambda p: (p.y, p.x)).tolist())  # Lat, Lon in radians
        data_for_clustering = coords
        
        # Override metric to haversine for geographic coordinates
        if metric == 'auto':
            effective_metric = 'haversine'
            results["params"]["metric"] = effective_metric
        elif metric != 'haversine':
            logger.warning(f"Using '{metric}' metric with geographic coordinates. 'haversine' is recommended.")
            
    elif 'vector_embedding' in features and len(features) == 1:
        # Special handling for vector embeddings
        using_vector_embedding = True
        logger.info("Using vector_embedding for clustering.")
        
        # Extract embeddings 
        embeddings = erratics_gdf['vector_embedding'].dropna().tolist()
        if not embeddings:
            logger.error("No valid vector embeddings found in data.")
            results["error"] = "No valid vector embeddings"
            return results
            
        # Convert list of embeddings to 2D numpy array
        try:
            data_for_clustering = np.array(embeddings)
            if len(data_for_clustering) < min_samples:
                logger.warning(f"Only {len(data_for_clustering)} erratics have valid embeddings (min_samples={min_samples}).")
                results["error"] = "Insufficient data points with valid embeddings"
                return results
                
            # Keep track of which rows in original df have valid embeddings for later ID mapping
            valid_indices = erratics_gdf['vector_embedding'].dropna().index
            valid_ids = erratics_gdf.loc[valid_indices, 'id']
            
            # Override metric to cosine for embeddings
            if metric == 'auto':
                effective_metric = 'cosine'
                results["params"]["metric"] = effective_metric
                
        except (ValueError, TypeError) as e:
            logger.error(f"Error preparing vector embeddings for clustering: {e}")
            results["error"] = f"Vector embedding preparation failed: {e}"
            return results
    else:
        # Regular feature columns
        logger.info(f"Using features {features} for clustering.")
        try:
            # Check all features exist in DataFrame
            missing = [f for f in features if f not in erratics_gdf.columns]
            if missing:
                raise KeyError(f"Features not found in erratic data: {missing}")
                
            # Extract features, drop rows with any NaN
            data_df = erratics_gdf[features].copy()
            data_df.dropna(inplace=True)
            if len(data_df) < min_samples:
                logger.warning(f"Only {len(data_df)} erratics have valid values for all required features.")
                results["error"] = "Insufficient data points with valid features"
                return results
                
            # Keep track of valid indices for ID mapping
            valid_indices = data_df.index
            valid_ids = erratics_gdf.loc[valid_indices, 'id']
            
            # Scale features
            scaler = StandardScaler()
            data_for_clustering = scaler.fit_transform(data_df)
            
            # Set metric to euclidean if auto
            if metric == 'auto':
                effective_metric = 'euclidean'
                results["params"]["metric"] = effective_metric
                
        except Exception as e:
            logger.error(f"Error preparing features for clustering: {e}")
            results["error"] = f"Feature preparation failed: {e}"
            return results

    # Estimate eps if not provided
    calculated_eps = eps
    if calculated_eps is None:
        # Estimate eps using k-distance graph
        k = min_samples * 2 # Factor used by NearestNeighbors 
        try:
            logger.info(f"Estimating eps using k-distance graph (k={k})...")
            nn = NearestNeighbors(n_neighbors=k, metric=effective_metric)
            nn.fit(data_for_clustering)
            distances, _ = nn.kneighbors(data_for_clustering)
            
            # Get the distance to the k-th nearest neighbor for each point
            k_distances = np.sort(distances[:, -1]) 
            
            # Option 1: Use kneed library if available
            try:
                from kneed import KneeLocator
                knee = KneeLocator(range(len(k_distances)), k_distances, curve='convex', direction='increasing')
                if knee.knee:
                    calculated_eps = k_distances[knee.knee]
                    logger.info(f"Estimated eps (knee point): {calculated_eps:.5f}")
                else: # Fallback if knee not found
                     # Use a percentile (e.g., 95th) or median as fallback heuristic
                     calculated_eps = np.percentile(k_distances, 90)
                     logger.info(f"Knee point not found, using 90th percentile eps: {calculated_eps:.5f}")

            except ImportError:
                 # Option 2: Fallback heuristic if kneed not available (e.g., median or percentile)
                 calculated_eps = np.percentile(k_distances, 90) 
                 logger.warning("kneed library not found for optimal eps estimation. Using 90th percentile heuristic.")
                 logger.info(f"Estimated eps (90th percentile): {calculated_eps:.5f}")

        except Exception as e:
            logger.error(f"Error estimating eps: {e}. Using default heuristic value.")
            # Provide a very rough default based on typical coordinate ranges if estimation fails
            if using_geographic_coords:
                calculated_eps = 0.01 # Radians, adjust based on expected data density
            elif using_vector_embedding and effective_metric == 'cosine':
                calculated_eps = 0.2  # Cosine distance range is 0-2
            else:
                calculated_eps = 0.5  # Generic default for standardized features

    results["params"]["eps"] = calculated_eps
    
    # Apply DBSCAN
    try:
        db = DBSCAN(eps=calculated_eps, min_samples=min_samples, metric=effective_metric)
        cluster_labels = db.fit_predict(data_for_clustering)
        
        # Map cluster labels to erratic IDs
        if using_geographic_coords:
            # All rows were used
            results["assignments"] = {int(erratics_gdf.iloc[i]['id']): int(label) for i, label in enumerate(cluster_labels)}
        else:
            # Only rows with valid features were used
            results["assignments"] = {int(valid_ids.iloc[i]): int(label) for i, label in enumerate(cluster_labels)}
        
        # Calculate metrics
        unique_labels = set(cluster_labels)
        results["num_clusters"] = len(unique_labels) - (1 if -1 in unique_labels else 0)
        results["num_noise_points"] = int(np.sum(cluster_labels == -1))
        
        logger.info(f"DBSCAN Result: Found {results['num_clusters']} clusters and {results['num_noise_points']} noise points.")

        # Calculate silhouette score if more than 1 cluster and fewer than n-1 points are noise
        n_samples = len(data_for_clustering)
        if results["num_clusters"] > 1 and n_samples - results["num_noise_points"] > results["num_clusters"]:
             # Need to exclude noise points for silhouette calculation
             try:
                 # Filter out noise points (-1 labels)
                 non_noise_mask = cluster_labels != -1
                 if np.sum(non_noise_mask) > results["num_clusters"]:  # Need more points than clusters
                     score = silhouette_score(
                         data_for_clustering[non_noise_mask], 
                         cluster_labels[non_noise_mask],
                         metric=effective_metric
                     )
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
        # Check if geometry-based features are requested
        use_geom_x = 'longitude' in features
        use_geom_y = 'latitude' in features
        is_vector_embedding_feature = features == ['vector_embedding']
        other_features = [f for f in features if f not in ['longitude', 'latitude', 'vector_embedding']]

        data_frames_to_concat = []
        ids_for_clustering = erratics_gdf['id'] # Default to all IDs
        data_for_clustering_df = None

        if is_vector_embedding_feature:
            logger.info("Using vector_embedding for K-Means clustering.")
            embeddings = erratics_gdf['vector_embedding'].dropna().tolist()
            if not embeddings or len(embeddings) < n_clusters:
                logger.error("Not enough valid vector embeddings for K-Means.")
                results["error"] = "Insufficient data points with valid embeddings"
                return results
            
            data_for_clustering_np = np.array(embeddings)
            valid_indices = erratics_gdf['vector_embedding'].dropna().index
            ids_for_clustering = erratics_gdf.loc[valid_indices, 'id']
            # K-Means typically uses Euclidean distance, scaling might still be beneficial for some embeddings
            # but for sentence transformers like all-MiniLM-L6-v2, they are often normalized to unit length,
            # making Euclidean distance related to cosine similarity. We'll scale by default.
            scaler = StandardScaler(with_mean=False) # Often embeddings are not mean-centered for scaling
            scaled_data = scaler.fit_transform(data_for_clustering_np)
        else:
            if use_geom_x:
                data_frames_to_concat.append(erratics_gdf.geometry.x.rename('longitude'))
            if use_geom_y:
                data_frames_to_concat.append(erratics_gdf.geometry.y.rename('latitude'))
            if other_features:
                missing_features = [f for f in other_features if f not in erratics_gdf.columns]
                if missing_features:
                    raise KeyError(f"Missing feature columns required for clustering: {missing_features}")
                data_frames_to_concat.append(erratics_gdf[other_features])

            if not data_frames_to_concat:
                 raise ValueError("No valid features selected for K-Means clustering.")
                 
            data_for_clustering_df = pd.concat(data_frames_to_concat, axis=1)
            data_for_clustering_df = data_for_clustering_df[features] # Reorder
           
            data_for_clustering_df.dropna(inplace=True) 
            ids_for_clustering = erratics_gdf.loc[data_for_clustering_df.index, 'id']

            if len(data_for_clustering_df) < n_clusters:
                 logger.warning(f"Not enough data points ({len(data_for_clustering_df)}) after handling NaNs for K-Means (k={n_clusters}).")
                 results["error"] = "Insufficient data points after NaN removal"
                 return results

            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(data_for_clustering_df)

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
        use_geom_x = 'longitude' in features
        use_geom_y = 'latitude' in features
        is_vector_embedding_feature = features == ['vector_embedding']
        other_features = [f for f in features if f not in ['longitude', 'latitude', 'vector_embedding']]

        data_frames_to_concat = []
        ids_for_clustering = erratics_gdf['id']
        scaled_data = None
        affinity = 'euclidean' # Default affinity

        if is_vector_embedding_feature:
            logger.info("Using vector_embedding for Hierarchical clustering.")
            embeddings = erratics_gdf['vector_embedding'].dropna().tolist()
            if not embeddings or len(embeddings) < n_clusters:
                logger.error("Not enough valid vector embeddings for Hierarchical Clustering.")
                results["error"] = "Insufficient data points with valid embeddings"
                return results
            
            data_for_clustering_np = np.array(embeddings)
            valid_indices = erratics_gdf['vector_embedding'].dropna().index
            ids_for_clustering = erratics_gdf.loc[valid_indices, 'id']
            
            # For hierarchical clustering, especially with cosine-like distances on embeddings,
            # using raw (but L2-normalized) embeddings directly with cosine affinity is common.
            # If sentence transformer provides normalized embeddings, no further scaling is strictly needed for cosine.
            # However, if linkage is 'ward', it requires euclidean.
            if linkage == 'ward':
                scaler = StandardScaler(with_mean=False) # Ward expects Euclidean, so scale like K-Means
                scaled_data = scaler.fit_transform(data_for_clustering_np)
                affinity = 'euclidean' 
            else:
                # For other linkages, if embeddings are normalized, cosine distance is good.
                # The AgglomerativeClustering affinity parameter can take 'cosine'.
                scaled_data = data_for_clustering_np # Use raw (hopefully normalized) embeddings
                affinity = 'cosine' 
            logger.info(f"Using affinity '{affinity}' for hierarchical clustering on embeddings.")

        else: # Geographic or other numeric features
            if use_geom_x:
                data_frames_to_concat.append(erratics_gdf.geometry.x.rename('longitude'))
            if use_geom_y:
                data_frames_to_concat.append(erratics_gdf.geometry.y.rename('latitude'))
            if other_features:
                missing_features = [f for f in other_features if f not in erratics_gdf.columns]
                if missing_features:
                    raise KeyError(f"Missing feature columns required for clustering: {missing_features}")
                data_frames_to_concat.append(erratics_gdf[other_features])

            if not data_frames_to_concat:
                 raise ValueError("No valid features selected for Hierarchical clustering.")
                 
            data_for_clustering_df = pd.concat(data_frames_to_concat, axis=1)
            data_for_clustering_df = data_for_clustering_df[features] # Reorder

            data_for_clustering_df.dropna(inplace=True)
            ids_for_clustering = erratics_gdf.loc[data_for_clustering_df.index, 'id']

            if len(data_for_clustering_df) < n_clusters:
                 logger.warning(f"Not enough data points ({len(data_for_clustering_df)}) after handling NaNs for Hierarchical Clustering (k={n_clusters}).")
                 results["error"] = "Insufficient data points after NaN removal"
                 return results

            # Determine affinity for geographic or other features
            if linkage != 'ward' and use_geom_x and use_geom_y and not other_features:
                # Only lat/lon specified, not ward linkage, use haversine (requires radians)
                coords_for_haversine = data_for_clustering_df[['latitude', 'longitude']].values
                scaled_data = np.radians(coords_for_haversine)
                affinity = 'haversine' 
                logger.info("Using 'haversine' affinity for geographic coordinates.")
            else:
                # Default to Euclidean for Ward or mixed/other features
                scaler = StandardScaler()
                scaled_data = scaler.fit_transform(data_for_clustering_df)
                affinity = 'euclidean'
                if linkage != 'ward':
                    logger.info("Using 'euclidean' affinity for mixed/other features or non-Ward linkage on scaled coordinates.")

        results["params"]["affinity_used"] = affinity # Log the actual affinity used

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
        # For Ward linkage, affinity must be 'euclidean' (already handled in data prep)
        # Other linkages can use different affinities/distance metrics if needed
        # The `affinity` variable is now set correctly based on feature type and linkage.
        
        model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage, affinity=affinity)
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
    parser.add_argument('--metric', type=str, default='auto', choices=['auto', 'haversine', 'euclidean', 'cosine'], 
                        help='DBSCAN: Distance metric to use. "auto" selects based on features.')
    
    # K-Means / Hierarchical args
    parser.add_argument('--k', type=int, default=5, help='K-Means/Hierarchical: Number of clusters.')
    parser.add_argument('--features', nargs='+', default=['longitude', 'latitude'], help='K-Means/Hierarchical/DBSCAN: Features to use for clustering.')
    parser.add_argument('--linkage', type=str, default='ward', choices=['ward', 'complete', 'average', 'single'], help='Hierarchical: Linkage criterion.')

    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        
    logger.info(f"Loading all erratic data for clustering...")
    erratics_gdf = load_erratics()
    
    # Debug: Print columns to check if 'geometry' exists
    if not erratics_gdf.empty:
        print("Columns in loaded GeoDataFrame:", erratics_gdf.columns)
        print("Active geometry column:", erratics_gdf.geometry.name)
    else:
        print("Loaded GeoDataFrame is empty.")
        
    if erratics_gdf.empty:
        logger.error("No erratics data loaded. Cannot perform clustering.")
        results = {"error": "No erratic data available"}
    elif args.algorithm == 'dbscan':
        results = perform_dbscan_clustering(erratics_gdf, eps=args.eps, min_samples=args.min_samples, 
                                            features=args.features, metric=args.metric)
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