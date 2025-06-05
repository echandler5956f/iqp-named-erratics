import sys
import pytest
from unittest import mock
import numpy as np
import geopandas as gpd
from shapely.geometry import Point
from sklearn.metrics import silhouette_score

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
def mock_sklearn_clustering_algos(tmp_path):
    """Mocks sklearn clustering algorithms where they are used in the clustering module."""
    mock_dbscan_instance = mock.MagicMock()
    # Default to 6 labels for mock_erratics_gdf_simple_coords, can be overridden in tests if needed
    mock_dbscan_instance.fit_predict.return_value = np.array([0, 0, 1, 1, 2, -1]) 

    mock_kmeans_instance = mock.MagicMock()
    mock_kmeans_instance.fit_predict.return_value = np.array([0, 0, 0, 1, 1, 1]) 
    mock_kmeans_instance.cluster_centers_ = np.array([[0.1, 0.1], [5.0, 5.0]]) 

    mock_agg_instance = mock.MagicMock()
    mock_agg_instance.fit_predict.return_value = np.array([0, 0, 0, 1, 1, 1])

    # Make fit_predict dynamic for more robust mocking if specific tests need varied input sizes
    def dynamic_fit_predict_dbscan(data):
        # Simple logic: create some clusters and noise based on data length
        n_samples = len(data)
        if n_samples == 6: # Specific case for the common test GDF
             return np.array([0, 0, 1, 1, 2, -1])
        labels = np.random.randint(0, max(1, n_samples // 3), size=n_samples)
        if n_samples > 4: labels[-1] = -1 # Add some noise for dbscan
        return labels
    mock_dbscan_instance.fit_predict.side_effect = dynamic_fit_predict_dbscan

    def dynamic_fit_predict_kmeans_agg(data):
        n_samples = len(data)
        if n_samples == 6: # Specific case for the common test GDF
            return np.array([0,0,0,1,1,1]) # e.g. two clusters
        return np.random.randint(0, max(1, n_samples // 2), size=n_samples)
    mock_kmeans_instance.fit_predict.side_effect = dynamic_fit_predict_kmeans_agg
    mock_agg_instance.fit_predict.side_effect = dynamic_fit_predict_kmeans_agg

    with mock.patch('clustering.DBSCAN', return_value=mock_dbscan_instance) as patched_dbscan, \
         mock.patch('clustering.KMeans', return_value=mock_kmeans_instance) as patched_kmeans, \
         mock.patch('clustering.AgglomerativeClustering', return_value=mock_agg_instance) as patched_agg:
        yield patched_dbscan, patched_kmeans, patched_agg

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

    @pytest.fixture
    def mock_kneed_locator_available(self):
        with mock.patch('clustering.KNEED_AVAILABLE', True):
            mock_klocator_instance = mock.MagicMock()
            mock_klocator_instance.knee = 3 # Example knee point
            with mock.patch('clustering.KneeLocator', return_value=mock_klocator_instance) as mock_klocator_class:
                yield mock_klocator_class, mock_klocator_instance

    @pytest.fixture
    def mock_kneed_locator_unavailable(self):
        with mock.patch('clustering.KNEED_AVAILABLE', False):
            yield None # KNEED_AVAILABLE is False, KneeLocator won't be called
    
    @pytest.fixture
    def mock_kneed_locator_no_knee(self):
        with mock.patch('clustering.KNEED_AVAILABLE', True):
            mock_klocator_instance = mock.MagicMock()
            mock_klocator_instance.knee = None # Simulate no knee found
            with mock.patch('clustering.KneeLocator', return_value=mock_klocator_instance) as mock_klocator_class:
                yield mock_klocator_class, mock_klocator_instance

    @mock.patch('clustering.NearestNeighbors')
    def test_estimate_dbscan_eps(self, MockNearestNeighbors, mock_kneed_locator_no_knee):
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
        
        # When mock_kneed_locator_no_knee is active, KneeLocator.knee is None,
        # so the function falls back to percentile calculation.
        # k_distances_in_func = np.sort(mock_distances[:, 1]) which is [0.05, 0.05, 0.05, 0.1, 0.1, 0.1]
        # expected_eps = np.percentile(k_distances_in_func, 90) which is 0.1
        if mock_kneed_locator_no_knee:
             k_distances_for_percentile = np.sort(mock_distances[:, 1]) # this is what the func calculates internally
             expected_fallback_eps = np.percentile(k_distances_for_percentile, 90)
             assert eps == expected_fallback_eps
        # else: # This branch would be for KNEED_AVAILABLE = False, not currently used by this specific test parameterization
             # This was the original problematic assertion structure
             # k_distances_sorted = np.sort(mock_distances[:,1])
             # assert eps == np.percentile(k_distances_sorted, 90)

    # Test for case where kneed IS available and DOES find a knee
    @mock.patch('clustering.NearestNeighbors')
    def test_estimate_dbscan_eps_with_knee(self, MockNearestNeighbors, mock_kneed_locator_available):
        mock_nn_instance = MockNearestNeighbors.return_value
        mock_distances = np.array([[0, 0.1], [0, 0.1], [0, 0.1], [0, 0.05], [0, 0.05], [0, 0.05]]) # simplified for min_samples=2 -> k=1
        mock_nn_instance.kneighbors.return_value = (mock_distances, None)
        
        sample_data = np.random.rand(6,2)
        # min_samples = 2, so k_distances are distances[:, 1] (distances to 1st neighbor)
        # If KneeLocator.knee = 3 (from fixture), and sorted k_distances are [0.05, 0.05, 0.05, 0.1, 0.1, 0.1]
        # then k_distances[3] = 0.1 is the expected eps.
        expected_eps_from_knee = np.sort(mock_distances[:,1])[mock_kneed_locator_available[1].knee]

        eps = _estimate_dbscan_eps(sample_data, min_samples=2, metric='euclidean', is_geo=False)
        assert eps is not None
        assert eps == expected_eps_from_knee

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
@mock.patch('utils.file_utils.json_to_file')
@mock.patch('clustering.perform_hierarchical_clustering')
@mock.patch('clustering.perform_kmeans_clustering')
@mock.patch('clustering.perform_dbscan_clustering')
@mock.patch('utils.db_utils.load_all_erratics_gdf')
class TestClusteringMain:
    def test_main_dbscan(self, mock_load_gdf, mock_dbscan, mock_kmeans, mock_hierarchical, mock_json_file, mock_erratics_gdf_simple_coords, tmp_path):
        mock_load_gdf.return_value = mock_erratics_gdf_simple_coords
        mock_args = mock.MagicMock()
        mock_args.algorithm = 'dbscan'
        mock_args.output = str(tmp_path / "dbscan_results.json")
        mock_args.eps = 0.1
        mock_args.min_samples = 3
        mock_args.features = ['longitude', 'latitude'] 
        mock_args.metric = 'haversine'
        mock_args.verbose = False

        mock_dbscan.return_value = {"algorithm": "DBSCAN", "assignments": {1:0}}

        mock_parser = mock.MagicMock()
        mock_parser.parse_args.return_value = mock_args
        with mock.patch('argparse.ArgumentParser', return_value=mock_parser):
            return_code = clustering_main()
        
        assert return_code == 0
        mock_dbscan.assert_called_once()
        mock_json_file.assert_called_once_with(mock_dbscan.return_value, str(tmp_path / "dbscan_results.json"))

    def test_main_kmeans_vector_features(self, mock_load_gdf, mock_dbscan, mock_kmeans, mock_hierarchical, mock_json_file, mock_erratics_gdf_with_embeddings, tmp_path):
        mock_load_gdf.return_value = mock_erratics_gdf_with_embeddings
        mock_args = mock.MagicMock()
        mock_args.algorithm = 'kmeans'
        mock_args.output = str(tmp_path / "kmeans_results.json")
        mock_args.k = 2
        mock_args.features = ['vector_embedding'] 
        mock_args.verbose = False

        mock_kmeans.return_value = {"algorithm": "KMeans", "assignments": {1:0}}

        mock_parser = mock.MagicMock()
        mock_parser.parse_args.return_value = mock_args
        with mock.patch('argparse.ArgumentParser', return_value=mock_parser):
            return_code = clustering_main()

        assert return_code == 0
        mock_kmeans.assert_called_once()
        mock_json_file.assert_called_once_with(mock_kmeans.return_value, str(tmp_path / "kmeans_results.json"))

    def test_main_error_unknown_algo(self, mock_load_gdf, mock_dbscan, mock_kmeans, mock_hierarchical, mock_json_file, mock_erratics_gdf_simple_coords, tmp_path):
        mock_load_gdf.return_value = mock_erratics_gdf_simple_coords
        mock_args = mock.MagicMock()
        mock_args.algorithm = 'unknown_algo'
        mock_args.output = str(tmp_path / "error.json")
        mock_args.verbose = False
        # Other args can use defaults from parser or be explicitly set on mock_args if needed by main before error

        mock_parser = mock.MagicMock()
        mock_parser.parse_args.return_value = mock_args
        with mock.patch('argparse.ArgumentParser', return_value=mock_parser):
            return_code = clustering_main()
        
        assert return_code == 1 # Expect error return code
        # Check that the output file still contains the error message
        expected_error_output = {"error": "Unknown algorithm: unknown_algo"}
        mock_json_file.assert_called_once_with(expected_error_output, str(tmp_path / "error.json"))

    def test_main_no_erratic_data(self, mock_load_gdf, mock_dbscan, mock_kmeans, mock_hierarchical, mock_json_file, tmp_path):
        mock_load_gdf.return_value = gpd.GeoDataFrame() # Empty GeoDataFrame
        mock_args = mock.MagicMock()
        mock_args.algorithm = 'dbscan' # Any valid algo
        mock_args.output = str(tmp_path / "no_data_error.json")
        mock_args.verbose = False

        mock_parser = mock.MagicMock()
        mock_parser.parse_args.return_value = mock_args
        with mock.patch('argparse.ArgumentParser', return_value=mock_parser):
            return_code = clustering_main()

        assert return_code == 1
        expected_error_output = {"error": "No erratic data available"}
        mock_json_file.assert_called_once_with(expected_error_output, str(tmp_path / "no_data_error.json")) 