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
from sklearn.neighbors import NearestNeighbors
try:
    from kneed import KneeLocator
    KNEED_AVAILABLE = True
except ImportError:
    KNEED_AVAILABLE = False
from sklearn.cluster import DBSCAN, KMeans, AgglomerativeClustering

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add the directory containing this script (backend/src/scripts/python) to sys.path
# This allows utils and data_pipeline to be imported as packages.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from utils import db_utils, file_utils

# --- Helper Functions ---

def _prepare_data_for_dbscan(
    erratics_gdf: gpd.GeoDataFrame, 
    features: typing.Optional[typing.List[str]], 
    min_samples: int, 
    metric_param: str
) -> typing.Tuple[typing.Optional[np.ndarray], typing.Optional[pd.Series], str, typing.Optional[str]]:
    """Prepares data for DBSCAN, determining features, metric, and handling scaling/transformations."""
    if erratics_gdf.empty or len(erratics_gdf) < min_samples:
        return None, None, metric_param, "Insufficient data points"

    data_for_clustering = None
    valid_ids = None
    effective_metric = metric_param
    error_message = None

    is_geo = features is None or (len(features) == 2 and any(f in features for f in ['longitude', 'latitude', 'lon', 'lat']))
    is_vector = features == ['vector_embedding']

    if is_geo:
        logger.info("Using geographic coordinates for DBSCAN.")
        coords = erratics_gdf.geometry.apply(lambda p: (p.y, p.x)).tolist()  # Lat, Lon
        data_for_clustering = np.radians(coords) # DBSCAN with haversine needs radians
        valid_ids = erratics_gdf['id']
        if metric_param == 'auto':
            effective_metric = 'haversine'
        elif metric_param != 'haversine':
            logger.warning(f"Using '{metric_param}' with geographic coordinates. 'haversine' is recommended for DBSCAN.")
    elif is_vector:
        logger.info("Using vector_embedding for DBSCAN.")
        embeddings = erratics_gdf['vector_embedding'].dropna().tolist()
        if not embeddings or len(embeddings) < min_samples:
            return None, None, effective_metric, "Insufficient data points with valid embeddings"
        
        try:
            data_for_clustering = np.array(embeddings)
            if data_for_clustering.ndim == 1: # Handle case of single embedding
                 if len(data_for_clustering) > 0 : # if embedding is not empty
                    data_for_clustering = data_for_clustering.reshape(1, -1)
                 else: # if embedding is empty
                    return None, None, effective_metric, "Empty vector embedding found"

            valid_indices = erratics_gdf['vector_embedding'].dropna().index
            valid_ids = erratics_gdf.loc[valid_indices, 'id']
            if metric_param == 'auto':
                effective_metric = 'cosine'
        except (ValueError, TypeError) as e:
            error_message = f"Vector embedding preparation failed: {e}"
            logger.error(error_message)
    else: # Other scalar features
        logger.info(f"Using features {features} for DBSCAN.")
        missing = [f for f in features if f not in erratics_gdf.columns]
        if missing:
            return None, None, effective_metric, f"Features not found: {missing}"
        
        data_df = erratics_gdf[features].copy()
        data_df.dropna(inplace=True)
        if len(data_df) < min_samples:
            return None, None, effective_metric, "Insufficient data points with valid features after NaN removal"

        valid_ids = erratics_gdf.loc[data_df.index, 'id']
        scaler = StandardScaler()
        data_for_clustering = scaler.fit_transform(data_df)
        if metric_param == 'auto':
            effective_metric = 'euclidean'
            
    if data_for_clustering is not None and len(data_for_clustering) < min_samples:
         error_message = "Insufficient data points after processing for DBSCAN."
         logger.warning(error_message)
         return None, None, effective_metric, error_message

    return data_for_clustering, valid_ids, effective_metric, error_message


def _estimate_dbscan_eps(data: np.ndarray, min_samples: int, metric: str, is_geo: bool) -> float:
    """Estimates the eps parameter for DBSCAN using the k-distance graph."""
    logger.info(f"Estimating eps using k-distance graph (k={min_samples * 2}, metric={metric})...")
    try:
        nn = NearestNeighbors(n_neighbors=min_samples * 2, metric=metric)
        nn.fit(data)
        distances, _ = nn.kneighbors(data)
        k_distances = np.sort(distances[:, min_samples -1]) # Distances to (min_samples-1)-th neighbor

        if KNEED_AVAILABLE:
            knee_locator = KneeLocator(range(len(k_distances)), k_distances, curve='convex', direction='increasing')
            if knee_locator.knee:
                estimated_eps = k_distances[knee_locator.knee]
                logger.info(f"Estimated eps (kneed): {estimated_eps:.5f}")
                return estimated_eps
            else:
                logger.warning("Knee point not found by kneed. Falling back to percentile.")
        
        # Fallback: percentile heuristic
        estimated_eps = np.percentile(k_distances, 90)
        logger.info(f"Estimated eps (90th percentile): {estimated_eps:.5f}")
        return estimated_eps

    except Exception as e:
        logger.error(f"Error estimating eps: {e}. Using default heuristic value.")
        if is_geo: return 0.01  # Radians, adjust based on expected data density
        if metric == 'cosine': return 0.2 # Cosine distance range 0-2
        return 0.5 # Generic for standardized features

def _prepare_data_for_kmeans_hierarchical(
    erratics_gdf: gpd.GeoDataFrame, 
    features: typing.List[str], 
    n_clusters: int, 
    algorithm: str, # 'kmeans' or 'hierarchical'
    linkage_method: typing.Optional[str] = 'ward' # For hierarchical
) -> typing.Tuple[typing.Optional[np.ndarray], typing.Optional[pd.Series], typing.Any, typing.Optional[str]]:
    """Prepares data for K-Means or Hierarchical clustering."""
    if erratics_gdf.empty or len(erratics_gdf) < n_clusters:
        return None, None, None, "Insufficient data points"

    data_to_process_df = None
    ids_for_clustering = erratics_gdf['id']
    affinity_or_scaler = 'euclidean' # Default for KMeans, or Hierarchical with Ward
    error_message = None
    
    is_vector_embedding = features == ['vector_embedding']

    if is_vector_embedding:
        logger.info(f"Using vector_embedding for {algorithm}.")
        embeddings = erratics_gdf['vector_embedding'].dropna().tolist()
        if not embeddings or len(embeddings) < n_clusters:
            return None, None, None, "Insufficient data points with valid embeddings"
        
        try:
            data_values = np.array(embeddings)
            if data_values.ndim == 1: data_values = data_values.reshape(-1,1) # Ensure 2D for single feature
            valid_indices = erratics_gdf['vector_embedding'].dropna().index
            ids_for_clustering = erratics_gdf.loc[valid_indices, 'id']

            if algorithm == 'kmeans':
                scaler = StandardScaler(with_mean=False) # Embeddings often not mean-centered
                processed_data = scaler.fit_transform(data_values)
                affinity_or_scaler = scaler
            elif algorithm == 'hierarchical':
                if linkage_method == 'ward': # Ward requires Euclidean
                    scaler = StandardScaler(with_mean=False)
                    processed_data = scaler.fit_transform(data_values)
                    affinity_or_scaler = 'euclidean'
                else: # Other linkages can use cosine directly with normalized embeddings
                    processed_data = data_values # Assume normalized
                    affinity_or_scaler = 'cosine'
            logger.info(f"Using affinity/metric '{affinity_or_scaler if isinstance(affinity_or_scaler, str) else 'scaled_euclidean'}' for {algorithm} on embeddings.")

        except (ValueError, TypeError) as e:
            error_message = f"Vector embedding preparation failed for {algorithm}: {e}"
            logger.error(error_message)
            return None, None, None, error_message
    else: # Geographic or other numeric features
        logger.info(f"Using features {features} for {algorithm}.")
        use_geom_x = 'longitude' in features
        use_geom_y = 'latitude' in features
        other_features = [f for f in features if f not in ['longitude', 'latitude']]

        temp_data_frames = []
        if use_geom_x: temp_data_frames.append(erratics_gdf.geometry.x.rename('longitude'))
        if use_geom_y: temp_data_frames.append(erratics_gdf.geometry.y.rename('latitude'))
        if other_features:
            missing = [f for f in other_features if f not in erratics_gdf.columns]
            if missing: return None, None, None, f"Missing features for {algorithm}: {missing}"
            temp_data_frames.append(erratics_gdf[other_features])
        
        if not temp_data_frames: return None, None, None, f"No valid features selected for {algorithm}."
        
        data_to_process_df = pd.concat(temp_data_frames, axis=1)
        data_to_process_df = data_to_process_df[features] # Ensure correct order
        data_to_process_df.dropna(inplace=True)
        ids_for_clustering = erratics_gdf.loc[data_to_process_df.index, 'id']

        if len(data_to_process_df) < n_clusters:
            return None, None, None, f"Insufficient data after NaN removal for {algorithm}"
        
        data_values = data_to_process_df.values

        if algorithm == 'kmeans':
            scaler = StandardScaler()
            processed_data = scaler.fit_transform(data_values)
            affinity_or_scaler = scaler
        elif algorithm == 'hierarchical':
            # For non-Ward on pure geo, consider Haversine (needs radians)
            if linkage_method != 'ward' and use_geom_x and use_geom_y and not other_features:
                processed_data = np.radians(data_values) # Expects [[lat, lon], ...]
                affinity_or_scaler = 'haversine'
            else: # Default to Euclidean for Ward or mixed/other features
                scaler = StandardScaler()
                processed_data = scaler.fit_transform(data_values)
                affinity_or_scaler = 'euclidean'
            logger.info(f"Using affinity '{affinity_or_scaler}' for hierarchical clustering.")
    
    if processed_data is not None and len(processed_data) < n_clusters:
        error_message = f"Insufficient data points after processing for {algorithm}."
        logger.warning(error_message)
        return None, None, affinity_or_scaler, error_message

    return processed_data, ids_for_clustering, affinity_or_scaler, error_message

def _calculate_silhouette_score_if_possible(
    data_points: np.ndarray, 
    labels: np.ndarray, 
    metric_for_score: str,
    num_actual_clusters: int, # Number of clusters excluding noise
    logger_instance: logging.Logger
) -> typing.Optional[float]:
    """Calculates silhouette score if conditions are met."""
    if num_actual_clusters <= 1:
        logger_instance.info("Silhouette score requires more than 1 cluster.")
        return None
    
    # Ensure enough distinct labels among non-noise points for silhouette score
    non_noise_labels = labels[labels != -1]
    if len(np.unique(non_noise_labels)) <=1:
        logger_instance.warning("Not enough distinct cluster labels among non-noise points for silhouette score.")
        return None

    if len(data_points) == 0 or len(labels) == 0 :
        logger_instance.warning("Empty data or labels for silhouette score.")
        return None
        
    # Silhouette score requires at least 2 labels and n_samples > n_labels
    # Also filter out noise points for DBSCAN-like results if present
    non_noise_mask = labels != -1
    if np.sum(non_noise_mask) > num_actual_clusters : # Must have more points than clusters after noise removal
        try:
            score = silhouette_score(
                data_points[non_noise_mask], 
                labels[non_noise_mask],
                metric=metric_for_score
            )
            logger_instance.info(f"Silhouette Score: {score:.3f}")
            return float(score)
        except ValueError as sil_err:
            logger_instance.warning(f"Could not calculate silhouette score: {sil_err}")
        except Exception as e:
            logger_instance.error(f"Unexpected error calculating silhouette score: {e}")
    else:
        logger_instance.warning("Not enough non-noise samples or clusters for silhouette score calculation.")
    return None

# --- Main Clustering Functions ---

def perform_dbscan_clustering(erratics_gdf: gpd.GeoDataFrame, eps: typing.Optional[float] = None, 
                              min_samples: int = 3, features: typing.Optional[typing.List[str]] = None,
                              metric: str = 'auto') -> typing.Dict:
    """Performs DBSCAN clustering on erratic coordinates or other features."""
    logger.info(f"Performing DBSCAN (min_samples={min_samples}, eps={eps or 'auto'}, metric='{metric}')...")
    results = {
        "algorithm": "DBSCAN",
        "params": {"min_samples": min_samples, "eps_param": eps, "features_param": features, "metric_param": metric},
        "assignments": {}, "num_clusters": 0, "num_noise_points": 0, "silhouette_score": None
    }

    data_for_clustering, valid_ids, effective_metric, error = _prepare_data_for_dbscan(
        erratics_gdf, features, min_samples, metric
    )
    results["params"]["metric_used"] = effective_metric
    if error:
        results["error"] = error
        logger.error(f"DBSCAN pre-processing failed: {error}")
        return results
    if data_for_clustering is None: # Should be caught by error, but as safeguard
        results["error"] = "Data preparation unexpectedly returned None"
        return results


    calculated_eps = eps
    if calculated_eps is None:
        is_geo_features = effective_metric == 'haversine'
        calculated_eps = _estimate_dbscan_eps(data_for_clustering, min_samples, effective_metric, is_geo_features)
    results["params"]["eps_calculated"] = calculated_eps
    
    try:
        db = DBSCAN(eps=calculated_eps, min_samples=min_samples, metric=effective_metric)
        cluster_labels = db.fit_predict(data_for_clustering)
        
        results["assignments"] = {int(valid_ids.iloc[i]): int(label) for i, label in enumerate(cluster_labels)}
        unique_labels = set(cluster_labels)
        results["num_clusters"] = len(unique_labels) - (1 if -1 in unique_labels else 0)
        results["num_noise_points"] = int(np.sum(cluster_labels == -1))
        logger.info(f"DBSCAN: {results['num_clusters']} clusters, {results['num_noise_points']} noise points.")

        results["silhouette_score"] = _calculate_silhouette_score_if_possible(
            data_for_clustering, cluster_labels, effective_metric, results["num_clusters"], logger
        )
    except Exception as e:
        logger.error(f"Error during DBSCAN execution: {e}", exc_info=True)
        results["error"] = f"DBSCAN execution failed: {e}"
    return results

def perform_kmeans_clustering(erratics_gdf: gpd.GeoDataFrame, n_clusters: int = 5, 
                              features: typing.List[str] = ['longitude', 'latitude']) -> typing.Dict:
    """Performs K-Means clustering on specified features."""
    logger.info(f"Performing K-Means (k={n_clusters}) using features: {features}...")
    results = {
        "algorithm": "KMeans",
        "params": {"n_clusters": n_clusters, "features": features},
        "assignments": {}, "silhouette_score": None, "cluster_centers": []
    }

    scaled_data, ids_for_clustering, scaler, error = _prepare_data_for_kmeans_hierarchical(
        erratics_gdf, features, n_clusters, 'kmeans'
    )
    if error:
        results["error"] = error; logger.error(f"K-Means pre-processing failed: {error}"); return results
    if scaled_data is None: results["error"] = "KMeans data prep returned None"; return results

    try:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
        cluster_labels = kmeans.fit_predict(scaled_data)
        
        results["assignments"] = {int(ids_for_clustering.iloc[i]): int(label) for i, label in enumerate(cluster_labels)}
        if hasattr(scaler, 'inverse_transform'): # Scaler object for K-Means
             results["cluster_centers"] = scaler.inverse_transform(kmeans.cluster_centers_).tolist()
        else: # Should not happen if scaler is returned by prep for KMeans
             results["cluster_centers"] = kmeans.cluster_centers_.tolist()


        results["silhouette_score"] = _calculate_silhouette_score_if_possible(
            scaled_data, cluster_labels, 'euclidean', n_clusters, logger # K-Means uses Euclidean on scaled data
        )
        logger.info(f"K-Means: Assigned {len(results['assignments'])} points to {n_clusters} clusters.")
    except Exception as e:
        logger.error(f"Error during K-Means execution: {e}", exc_info=True)
        results["error"] = f"K-Means execution failed: {e}"
    return results

def perform_hierarchical_clustering(erratics_gdf: gpd.GeoDataFrame, n_clusters: int = 5, 
                                    linkage: str = 'ward', 
                                    features: typing.List[str] = ['longitude', 'latitude']) -> typing.Dict:
    """Performs Agglomerative Hierarchical clustering."""
    logger.info(f"Performing Hierarchical (k={n_clusters}, linkage={linkage}) using features: {features}...")
    results = {
        "algorithm": "Hierarchical",
        "params": {"n_clusters": n_clusters, "linkage": linkage, "features": features},
        "assignments": {}, "silhouette_score": None
    }

    processed_data, ids_for_clustering, affinity_used, error = _prepare_data_for_kmeans_hierarchical(
        erratics_gdf, features, n_clusters, 'hierarchical', linkage_method=linkage
    )
    results["params"]["affinity_used"] = affinity_used 
    if error:
        results["error"] = error; logger.error(f"Hierarchical pre-processing failed: {error}"); return results
    if processed_data is None: results["error"] = "Hierarchical data prep returned None"; return results
    
    # Ward linkage requires affinity to be 'euclidean'. Other linkages can use other affinities.
    # The _prepare_data function already ensures processed_data is suitable for the affinity.
    if linkage == 'ward' and affinity_used != 'euclidean':
        logger.warning(f"Ward linkage specified, but affinity is {affinity_used}. Forcing Euclidean on processed data.")
        # This case implies _prepare_data_for_kmeans_hierarchical might need refinement or this is a safeguard
        # For now, assume data is already scaled if ward was intended, so affinity can be set to Euclidean.
        # If data was prepared for 'haversine' (e.g. geo + non-ward), but then 'ward' is passed here, it's an issue.
        # The design of _prepare_data_for_kmeans_hierarchical tries to provide correct data for the (linkage, feature_type) combo.
        affinity_for_model = 'euclidean' # Ward linkage must use Euclidean distance.
    else:
        affinity_for_model = affinity_used

    try:
        model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage, affinity=affinity_for_model)
        cluster_labels = model.fit_predict(processed_data)
        
        results["assignments"] = {int(ids_for_clustering.iloc[i]): int(label) for i, label in enumerate(cluster_labels)}
        
        # Metric for silhouette score should match the affinity used if possible
        metric_for_silhouette = affinity_used if affinity_used != 'precomputed' else 'euclidean'
        results["silhouette_score"] = _calculate_silhouette_score_if_possible(
            processed_data, cluster_labels, metric_for_silhouette, n_clusters, logger
        )
        logger.info(f"Hierarchical: Assigned {len(results['assignments'])} points to {n_clusters} clusters.")
    except Exception as e:
        logger.error(f"Error during Hierarchical execution: {e}", exc_info=True)
        results["error"] = f"Hierarchical execution failed: {e}"
    return results

# --- Main Execution ---

def main():
    parser = argparse.ArgumentParser(description='Perform clustering on glacial erratics.')
    parser.add_argument('--algorithm', type=str, required=True, choices=['dbscan', 'kmeans', 'hierarchical'], help='Clustering algorithm to use.')
    parser.add_argument('--output', type=str, required=True, help='Output file for results (JSON)')
    
    # DBSCAN args
    parser.add_argument('--eps', type=float, help='DBSCAN: Epsilon parameter. Auto-estimated if not provided.')
    parser.add_argument('--min_samples', type=int, default=3, help='DBSCAN: Minimum number of samples.')
    parser.add_argument('--metric', type=str, default='auto', choices=['auto', 'haversine', 'euclidean', 'cosine'], 
                        help='DBSCAN: Distance metric. "auto" selects based on features.')
    
    # K-Means / Hierarchical args
    parser.add_argument('--k', type=int, default=5, help='K-Means/Hierarchical: Number of clusters.')
    parser.add_argument('--features', nargs='+', default=['longitude', 'latitude'], help='Features for clustering (e.g., longitude latitude, vector_embedding, or other column names).')
    parser.add_argument('--linkage', type=str, default='ward', choices=['ward', 'complete', 'average', 'single'], help='Hierarchical: Linkage criterion.')

    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG) # Set root logger to DEBUG
        logger.setLevel(logging.DEBUG) # Ensure this module's logger is also DEBUG
        
    logger.info(f"Loading all erratic data for clustering...")
    # Ensure NLTK path for db_utils if it uses text features indirectly, though unlikely for load_all_erratics_gdf
    # However, db_utils itself handles .env loading.

    erratics_gdf = db_utils.load_all_erratics_gdf() # This now uses the refactored db_utils
        
    if erratics_gdf.empty:
        logger.error("No erratics data loaded. Cannot perform clustering.")
        results = {"error": "No erratic data available"}
    elif args.algorithm == 'dbscan':
        # Ensure features is a list if provided, or None
        features_list = args.features if isinstance(args.features, list) else None
        if features_list and len(features_list) == 1 and ',' in features_list[0]: # Handle space-separated string from argparse
            features_list = [f.strip() for f in features_list[0].split(',')]

        results = perform_dbscan_clustering(erratics_gdf, eps=args.eps, min_samples=args.min_samples, 
                                            features=features_list, metric=args.metric)
    elif args.algorithm == 'kmeans':
        results = perform_kmeans_clustering(erratics_gdf, n_clusters=args.k, features=args.features)
    elif args.algorithm == 'hierarchical':
        results = perform_hierarchical_clustering(erratics_gdf, n_clusters=args.k, linkage=args.linkage, features=args.features)
    else:
        results = {"error": f"Unknown algorithm: {args.algorithm}"}

    logger.info(f"Writing clustering results to {args.output}")
    file_utils.json_to_file(results, args.output) # This now uses the refactored file_utils
    
    if 'error' in results:
        logger.error(f"Clustering failed: {results['error']}")
    else:
        logger.info("Clustering analysis complete.")
    
    print(json.dumps(results, indent=2)) # Print results to stdout for Node.js to capture
    
    return 0 if 'error' not in results else 1

if __name__ == "__main__":
    # Ensure utils can be found when run directly
    current_file_dir = os.path.dirname(os.path.abspath(__file__))
    # If 'utils' is a sibling directory to the 'python' directory (where this script is)
    # then we might need to add parent of current_file_dir
    # However, the current setup adds SCRIPT_DIR (i.e. .../python/) to path,
    # and utils is .../python/utils/, so `from utils import ...` should work.
    if current_file_dir not in sys.path: # This was SCRIPT_DIR before, should be consistent
        sys.path.insert(0, current_file_dir)
    
    # Re-import for safety, though top-level imports should be fine with current sys.path logic
    from utils import db_utils, file_utils 
    
    sys.exit(main()) 