import sys
import pytest
from unittest import mock
import numpy as np
import geopandas as gpd
from shapely.geometry import Point

# Module to test
from clustering import (
    perform_dbscan_clustering, 
    perform_kmeans_clustering, 
    perform_hierarchical_clustering,
    _prepare_data_for_dbscan, # Testing helper directly
    _estimate_dbscan_eps,     # Testing helper directly
    _prepare_data_for_kmeans_hierarchical, # Testing helper directly
    _calculate_silhouette_score_if_possible, # Testing helper directly
    main as clustering_main
)

# --- Fixtures ---
@pytest.fixture
def mock_erratics_gdf_simple_coords():
    return gpd.GeoDataFrame({
        'id': [1, 2, 3, 4, 5, 6],
        'name': [f'E{i}' for i in range(1,7)],
        'geometry': [Point(0,0), Point(0.1,0.1), Point(0.2,0.2), Point(5,5), Point(5.1,5.1), Point(5.2,5.2)]
    }, crs="EPSG:4326")

@pytest.fixture
def mock_erratics_gdf_with_embeddings():
    # Ensure embeddings are lists, not np.arrays, as GDF might store them that way before processing
    embeddings = [
        [0.1, 0.2, 0.3], [0.11, 0.21, 0.31], [0.12, 0.22, 0.32],
        [0.8, 0.9, 1.0], [0.81, 0.91, 1.01], [0.82, 0.92, 1.02]
    ]
    return gpd.GeoDataFrame({
        'id': [1,2,3,4,5,6],
        'name': [f'E{i}' for i in range(1,7)],
        'vector_embedding': embeddings,
        'geometry': [Point(i,i) for i in range(6)] # Dummy geometry
    }, crs="EPSG:4326")

@pytest.fixture
def mock_erratics_gdf_scalar_features():
    return gpd.GeoDataFrame({
        'id': [1,2,3,4,5,6],
        'feat1': [10, 11, 12, 100, 101, 102],
        'feat2': [5, 6, 7, 50, 51, 52],
        'geometry': [Point(i,i) for i in range(6)]
    }, crs="EPSG:4326")

# --- Mocks for Patches ---
@pytest.fixture
def mock_db_load_all_gdf(mock_erratics_gdf_simple_coords):
    # Default to simple coords, can be changed per test
    with mock.patch('utils.db_utils.load_all_erratics_gdf') as mock_load:
        mock_load.return_value = mock_erratics_gdf_simple_coords
        yield mock_load

@pytest.fixture
def mock_sklearn_clustering_algos():
    with mock.patch('sklearn.cluster.DBSCAN') as MockDBSCAN, \
         mock.patch('sklearn.cluster.KMeans') as MockKMeans, \
         mock.patch('sklearn.cluster.AgglomerativeClustering') as MockAgglomerative: 
        # Configure default mock behavior if needed, e.g., fit_predict returns labels
        MockDBSCAN.return_value.fit_predict.return_value = np.array([0,0,0,1,1,1])
        MockKMeans.return_value.fit_predict.return_value = np.array([0,0,0,1,1,1])
        MockKMeans.return_value.cluster_centers_ = np.array([[0.1,0.1],[5.1,5.1]])
        MockAgglomerative.return_value.fit_predict.return_value = np.array([0,0,0,1,1,1])
        yield MockDBSCAN, MockKMeans, MockAgglomerative

@pytest.fixture
def mock_kneed_locator(monkeypatch):
    if 'clustering.KNEED_AVAILABLE' in sys.modules['clustering'].__dict__ and sys.modules['clustering'].KNEED_AVAILABLE:
        mock_knee = mock.MagicMock()
        mock_knee.knee = 3 # Simulate finding a knee point index
        with mock.patch('kneed.KneeLocator', return_value=mock_knee) as patched_kneed:
            yield patched_kneed
    else:
        yield None # Kneed not available, no need to patch

# --- Test Helper Functions --- 
class TestClusteringHelpers:
    def test_prepare_data_for_dbscan_geo(self, mock_erratics_gdf_simple_coords):
        data, ids, metric, err = _prepare_data_for_dbscan(mock_erratics_gdf_simple_coords, None, 3, 'auto')
        assert err is None
        assert data is not None
        assert data.shape == (6,2) # 6 points, 2 coords (lat, lon)
        assert metric == 'haversine'
        assert len(ids) == 6

    def test_prepare_data_for_dbscan_vector(self, mock_erratics_gdf_with_embeddings):
        data, ids, metric, err = _prepare_data_for_dbscan(mock_erratics_gdf_with_embeddings, ['vector_embedding'], 3, 'auto')
        assert err is None
        assert data is not None
        assert data.shape == (6,3) # 6 points, 3 embedding dims
        assert metric == 'cosine'

    @mock.patch('sklearn.neighbors.NearestNeighbors')
    def test_estimate_dbscan_eps(self, MockNearestNeighbors, mock_kneed_locator):
        # Mock NearestNeighbors behavior
        mock_nn_instance = MockNearestNeighbors.return_value
        # 6 points, simulate distances to (min_samples-1)=1st neighbor for min_samples=2 (k=min_samples-1)
        # Here min_samples=2 (passed to NearestNeighbors is min_samples*2, but distances are for min_samples-1)
        # distances should be shape (n_samples, n_neighbors_func_arg), so (6, 2*2=4)
        # k_distances = distances[:, min_samples-1] = distances[:, 1]
        mock_distances = np.array([[0, 0.1, 0.2, 0.3], [0, 0.1, 0.2,0.3], [0, 0.1,0.2,0.3], 
                                   [0, 0.05,0.1,0.15], [0,0.05,0.1,0.15], [0,0.05,0.1,0.15]])
        mock_nn_instance.kneighbors.return_value = (mock_distances, None) # distances, indices
        
        sample_data = np.random.rand(6,2) # Dummy data for fitting
        eps = _estimate_dbscan_eps(sample_data, min_samples=2, metric='euclidean', is_geo=False)
        assert eps is not None
        # If kneed was patched and mock_knee.knee=3, then sorted_k_distances[3] would be the value
        # k_distances = np.sort(mock_distances[:, 1]) -> [0.05, 0.05, 0.05, 0.1, 0.1, 0.1]
        # if mock_kneed_locator.knee = 3, eps = k_distances[3] = 0.1
        if mock_kneed_locator: # Kneed was available and patched
             assert eps == mock_distances[:,1][3] # Based on mock_knee.knee = 3 and sorted distances
        else: # Fallback to percentile
             assert eps == np.percentile(np.sort(mock_distances[:,1]), 90)

# --- Test Main Clustering Functions ---
class TestClusteringAlgorithms:
    def test_perform_dbscan(self, mock_db_load_all_gdf, mock_sklearn_clustering_algos, mock_erratics_gdf_simple_coords):
        MockDBSCAN, _, _ = mock_sklearn_clustering_algos
        mock_db_load_all_gdf.return_value = mock_erratics_gdf_simple_coords # Ensure correct GDF is loaded
        
        results = perform_dbscan_clustering(mock_erratics_gdf_simple_coords, eps=0.5, min_samples=2, features=None, metric='haversine')
        assert 'error' not in results
        assert results['algorithm'] == "DBSCAN"
        assert len(results['assignments']) == 6
        assert results['num_clusters'] > 0 # Based on mock DBSCAN output
        MockDBSCAN.assert_called_once()
        # Check if DBSCAN was called with radians for geo coordinates
        # The data passed to DBSCAN().fit_predict() should be in radians
        # Actual data passed: np.radians(coords from GDF)
        # Call args for fit_predict: first arg is data
        # call_args[0][0] is the first positional argument to fit_predict
        assert MockDBSCAN.return_value.fit_predict.call_args[0][0].max() < (2 * np.pi) 

    def test_perform_kmeans(self, mock_db_load_all_gdf, mock_sklearn_clustering_algos, mock_erratics_gdf_simple_coords):
        _, MockKMeans, _ = mock_sklearn_clustering_algos
        mock_db_load_all_gdf.return_value = mock_erratics_gdf_simple_coords

        results = perform_kmeans_clustering(mock_erratics_gdf_simple_coords, n_clusters=2, features=['longitude', 'latitude'])
        assert 'error' not in results
        assert results['algorithm'] == "KMeans"
        assert len(results['assignments']) == 6
        assert len(results['cluster_centers']) == 2
        MockKMeans.assert_called_once_with(n_clusters=2, random_state=42, n_init='auto')

    def test_perform_hierarchical(self, mock_db_load_all_gdf, mock_sklearn_clustering_algos, mock_erratics_gdf_simple_coords):
        _, _, MockAgglomerative = mock_sklearn_clustering_algos
        mock_db_load_all_gdf.return_value = mock_erratics_gdf_simple_coords

        results = perform_hierarchical_clustering(mock_erratics_gdf_simple_coords, n_clusters=2, linkage='ward', features=['longitude', 'latitude'])
        assert 'error' not in results
        assert results['algorithm'] == "Hierarchical"
        assert len(results['assignments']) == 6
        MockAgglomerative.assert_called_once_with(n_clusters=2, linkage='ward', affinity='euclidean')

# --- Test Main CLI ---
@mock.patch('clustering.perform_dbscan_clustering')
@mock.patch('clustering.perform_kmeans_clustering')
@mock.patch('clustering.perform_hierarchical_clustering')
@mock.patch('utils.file_utils.json_to_file')
class TestClusteringMain:
    def test_main_dbscan(self, mock_json_file, mock_hierarchical, mock_kmeans, mock_dbscan, mock_db_load_all_gdf):
        mock_args = mock.MagicMock()
        mock_args.algorithm = 'dbscan'
        mock_args.output = "dbscan_results.json"
        mock_args.eps = 0.1
        mock_args.min_samples = 3
        mock_args.features = ['longitude', 'latitude'] # argparse default
        mock_args.metric = 'haversine'
        mock_args.verbose = False
        
        mock_dbscan.return_value = {"algorithm": "DBSCAN", "assignments": {1:0}}

        with mock.patch('argparse.ArgumentParser.parse_args', return_value=mock_args):
            with mock.patch('sys.exit') as mock_sys_exit:
                clustering_main()
                mock_sys_exit.assert_called_once_with(0)
        
        mock_db_load_all_gdf.assert_called_once()
        mock_dbscan.assert_called_once_with(
            mock_db_load_all_gdf.return_value, 
            eps=0.1, min_samples=3, features=['longitude', 'latitude'], metric='haversine'
        )
        mock_json_file.assert_called_once()

    def test_main_kmeans_vector_features(self, mock_json_file, mock_hierarchical, mock_kmeans, mock_dbscan, mock_db_load_all_gdf, mock_erratics_gdf_with_embeddings):
        mock_db_load_all_gdf.return_value = mock_erratics_gdf_with_embeddings
        mock_args = mock.MagicMock()
        mock_args.algorithm = 'kmeans'
        mock_args.output = "kmeans_results.json"
        mock_args.k = 2
        mock_args.features = ['vector_embedding'] # Specific feature
        mock_args.verbose = False
        
        mock_kmeans.return_value = {"algorithm": "KMeans", "assignments": {1:0}}

        with mock.patch('argparse.ArgumentParser.parse_args', return_value=mock_args):
            with mock.patch('sys.exit') as mock_sys_exit:
                clustering_main()
                mock_sys_exit.assert_called_once_with(0)

        mock_kmeans.assert_called_once_with(
            mock_erratics_gdf_with_embeddings, 
            n_clusters=2, features=['vector_embedding']
        )
        mock_json_file.assert_called_once()

    def test_main_error_unknown_algo(self, mock_json_file, mock_hierarchical, mock_kmeans, mock_dbscan, mock_db_load_all_gdf):
        mock_args = mock.MagicMock()
        mock_args.algorithm = 'unknown_algo'
        mock_args.output = "error.json"
        # ... other args defaults ...

        with mock.patch('argparse.ArgumentParser.parse_args', return_value=mock_args):
            with mock.patch('sys.exit') as mock_sys_exit:
                clustering_main()
                mock_sys_exit.assert_called_once_with(1) # Error exit code
        
        mock_json_file.assert_called_once_with({"error": "Unknown algorithm: unknown_algo"}, "error.json") 