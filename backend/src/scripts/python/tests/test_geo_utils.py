import pytest
from unittest import mock
import numpy as np
import geopandas as gpd
from shapely.geometry import Point as ShapelyPoint
import rasterio

# Modules to test
from utils import geo_utils
from utils.geo_utils import Point # Explicitly import Point for use in tests
from data_pipeline.sources import DataSource # For mocking in load_dem_data

# --- Fixtures ---
@pytest.fixture
def sample_point_a():
    return Point(longitude=-75.0, latitude=45.0)

@pytest.fixture
def sample_point_b():
    return Point(longitude=-70.0, latitude=40.0)

@pytest.fixture
def sample_features_gdf(sample_point_a, sample_point_b):
    data = {
        'id': [1, 2, 3],
        'name': ['Feat1', 'Feat2', 'Feat3'],
        'geometry': [sample_point_a.to_shapely(), sample_point_b.to_shapely(), ShapelyPoint(-72, 42)]
    }
    return gpd.GeoDataFrame(data, crs="EPSG:4326")

@pytest.fixture
def mock_rasterio_open_with_data(monkeypatch):
    """Mocks rasterio.open to return a mock dataset object."""
    mock_dataset = mock.MagicMock(spec=rasterio.io.DatasetReader)
    mock_dataset.crs = rasterio.crs.CRS.from_epsg(4326)
    mock_dataset.nodata = -9999
    mock_dataset.res = (0.00027777, 0.00027777) # approx 30m for geographic
    mock_dataset.height = 100
    mock_dataset.width = 100
    mock_dataset.transform = rasterio.Affine(0.1, 0, -180, 0, -0.1, 90) # Example transform

    def sample_data_generator(coords_list):
        # Simulate returning elevation based on y-coordinate for simplicity
        for x, y in coords_list:
            yield np.array([y * 10]) # Dummy elevation value
    mock_dataset.sample.side_effect = sample_data_generator

    def read_data_window(band_index, window=None):
        # Return a fixed array for window reads, e.g., for TRI or landscape metrics
        # This needs to be adjusted based on what the test expects
        if window and window.width == 3 and window.height == 3:
             # For TRI, return a 3x3 array
            return np.array([[10,12,11],[15,20,18],[19,22,21]], dtype=np.float32)
        return np.full((window.height if window else 10, window.width if window else 10), 100, dtype=np.float32)
    mock_dataset.read.side_effect = read_data_window
    
    # mock_dataset.index method simulation
    def mock_index(lon, lat):
        # Simplified: assume point is within bounds for mock, return dummy indices
        if -180 <= lon <= 180 and -90 <= lat <= 90:
            return (50,50) # Dummy row, col
        raise rasterio.errors.RasterioIOError("Point outside bounds")
    mock_dataset.index = mock_index

    mock_open = mock.MagicMock(return_value=mock_dataset)
    # The context manager part: __enter__ returns the mock_dataset
    mock_open.return_value.__enter__.return_value = mock_dataset
    monkeypatch.setattr(rasterio, 'open', mock_open)
    return mock_open, mock_dataset

# --- Test Classes ---

class TestPointClass:
    def test_point_creation(self):
        p = Point(longitude=-70.5, latitude=42.3)
        assert p.longitude == -70.5
        assert p.latitude == 42.3

    def test_point_to_shapely(self):
        p = Point(-70.5, 42.3)
        sp = p.to_shapely()
        assert isinstance(sp, ShapelyPoint)
        assert sp.x == -70.5
        assert sp.y == 42.3

    def test_point_to_tuple(self):
        p = Point(-70.5, 42.3)
        assert p.to_tuple() == (-70.5, 42.3)

class TestHaversineDistance:
    def test_haversine_known_distance(self):
        # Boston to New York (approximate)
        lat1, lon1 = 42.3601, -71.0589  # Boston
        lat2, lon2 = 40.7128, -74.0060  # New York
        # Expected distance ~305-306 km
        dist = geo_utils.haversine_distance(lat1, lon1, lat2, lon2)
        assert 305000 < dist < 307000

    def test_haversine_zero_distance(self):
        lat1, lon1 = 40.0, -70.0
        assert geo_utils.haversine_distance(lat1, lon1, lat1, lon1) == 0.0

class TestElevationCategorization:
    @pytest.mark.parametrize("elevation, expected_category", [
        (-10, "below_sea_level"), (0, "lowland"), (199, "lowland"),
        (200, "upland"), (499, "upland"), (500, "hill"), (999, "hill"),
        (1000, "low_mountain"), (1999, "low_mountain"), (2000, "mid_mountain"),
        (2999, "mid_mountain"), (3000, "high_mountain"), (4999, "high_mountain"),
        (5000, "extreme_elevation")
    ])
    def test_categorize_elevation(self, elevation, expected_category):
        assert geo_utils.categorize_elevation(elevation) == expected_category
        assert geo_utils.get_elevation_category(elevation) == expected_category

class TestLoadDemData:
    @mock.patch('utils.geo_utils.GlobalDataRegistry') # Patch where it's used
    @mock.patch('os.path.exists')
    def test_load_dem_data_success(self, mock_os_exists, mock_global_registry, sample_point_a, tmp_path):
        mock_os_exists.return_value = True # Assume tile path exists
        
        tile_path_1 = str(tmp_path / "tile1.tif")
        (tmp_path / "tile1.tif").touch()
        
        mock_gmted_source = DataSource(
            name="gmted_elevation_tiled", source_type="file", format="geotiff",
            is_tiled=True,
            tile_paths=[tile_path_1, str(tmp_path / "tile2.tif")],
            tile_centers=[(-75.0, 45.0), (0.0, 0.0)], # Center of tile1 covers sample_point_a
            tile_size_degrees=30.0
        )
        mock_global_registry.get_all_sources.return_value = {"gmted_elevation_tiled": mock_gmted_source}

        result_path = geo_utils.load_dem_data(sample_point_a)
        assert result_path == tile_path_1
        mock_os_exists.assert_called_with(tile_path_1)

    @mock.patch('utils.geo_utils.GlobalDataRegistry')
    def test_load_dem_data_point_outside_all_tiles(self, mock_global_registry, tmp_path):
        tile_path_1 = str(tmp_path / "tile_far.tif")
        mock_gmted_source = DataSource(
            name="gmted_elevation_tiled", source_type="file", format="geotiff",
            is_tiled=True,
            tile_paths=[tile_path_1],
            tile_centers=[(0.0, 0.0)], # Tile centered at equator/prime meridian
            tile_size_degrees=30.0
        )
        mock_global_registry.get_all_sources.return_value = {"gmted_elevation_tiled": mock_gmted_source}
        
        far_point = Point(longitude=100.0, latitude=80.0)
        assert geo_utils.load_dem_data(far_point) is None

    @mock.patch('utils.geo_utils.GlobalDataRegistry')
    @mock.patch('os.path.exists', return_value=False) # Tile path does not exist
    def test_load_dem_data_tile_path_not_exists(self, mock_os_exists, mock_global_registry, sample_point_a, tmp_path):
        tile_path_1 = str(tmp_path / "non_existent_tile.tif")
        # Do NOT create the file with .touch()
        mock_gmted_source = DataSource(
            name="gmted_elevation_tiled", source_type="file", format="geotiff",
            is_tiled=True,
            tile_paths=[tile_path_1],
            tile_centers=[(-75.0, 45.0)],
            tile_size_degrees=30.0
        )
        mock_global_registry.get_all_sources.return_value = {"gmted_elevation_tiled": mock_gmted_source}
        assert geo_utils.load_dem_data(sample_point_a) is None 
        # os.path.exists is called inside load_dem_data for the resolved path
        mock_os_exists.assert_called_with(tile_path_1)

    @mock.patch('utils.geo_utils.GlobalDataRegistry')
    def test_load_dem_data_source_misconfigured(self, mock_global_registry, sample_point_a):
        mock_gmted_source_bad = DataSource(
            name="gmted_elevation_tiled", source_type="file", format="geotiff",
            is_tiled=True, tile_paths=[], tile_centers=[], tile_size_degrees=None # Missing essential info
        )
        mock_global_registry.get_all_sources.return_value = {"gmted_elevation_tiled": mock_gmted_source_bad}
        assert geo_utils.load_dem_data(sample_point_a) is None

    @mock.patch('utils.geo_utils.PIPELINE_AVAILABLE', False)
    def test_load_dem_data_pipeline_not_available(self, sample_point_a):
        assert geo_utils.load_dem_data(sample_point_a) is None

    def test_load_dem_data_no_point(self):
        assert geo_utils.load_dem_data(None) is None

class TestGetElevationAtPoint:
    def test_get_elevation_success(self, mock_rasterio_open_with_data, sample_point_a, tmp_path):
        dem_file = tmp_path / "dummy_dem.tif"
        dem_file.touch() # Needs to exist for rasterio.open mock path validation if any
        mock_open, mock_ds = mock_rasterio_open_with_data
        
        elevation = geo_utils.get_elevation_at_point(sample_point_a, str(dem_file))
        assert elevation == sample_point_a.latitude * 10 # As per mock_dataset.sample logic
        mock_open.assert_called_once_with(str(dem_file))
        mock_ds.sample.assert_called_once_with([(sample_point_a.longitude, sample_point_a.latitude)])

    def test_get_elevation_on_nodata(self, mock_rasterio_open_with_data, sample_point_a, tmp_path):
        dem_file = tmp_path / "dummy_dem_nodata.tif"
        dem_file.touch()
        mock_open, mock_ds = mock_rasterio_open_with_data
        mock_ds.sample.side_effect = lambda coords_list: iter([np.array([mock_ds.nodata])])

        elevation = geo_utils.get_elevation_at_point(sample_point_a, str(dem_file))
        assert elevation is None

    @mock.patch('utils.geo_utils.RASTERIO_AVAILABLE', False)
    def test_get_elevation_rasterio_not_available(self, sample_point_a):
        assert geo_utils.get_elevation_at_point(sample_point_a, "dummy.tif") is None

    def test_get_elevation_dem_not_found(self, sample_point_a):
        assert geo_utils.get_elevation_at_point(sample_point_a, "/non/existent/dem.tif") is None

class TestFindNearestFeature:
    def test_find_nearest_success(self, sample_point_a, sample_features_gdf):
        # sample_point_a is one of the features, so distance should be 0
        feature_data, distance = geo_utils.find_nearest_feature(sample_point_a, sample_features_gdf)
        assert distance == 0.0
        assert feature_data is not None
        assert feature_data['id'] == 1
        assert feature_data['name'] == 'Feat1'

    def test_find_nearest_different_point(self, sample_features_gdf):
        test_point = Point(longitude=-74.9, latitude=44.9) # Close to sample_point_a
        feature_data, distance = geo_utils.find_nearest_feature(test_point, sample_features_gdf)
        assert distance > 0
        assert feature_data['id'] == 1 # Should still be Feat1 (sample_point_a)

    def test_find_nearest_empty_gdf(self, sample_point_a):
        empty_gdf = gpd.GeoDataFrame(columns=['id', 'geometry'], geometry='geometry', crs="EPSG:4326")
        feature_data, distance = geo_utils.find_nearest_feature(sample_point_a, empty_gdf)
        assert feature_data is None
        assert distance == float('inf')

class TestLandscapeAndGeomorph:
    def test_calculate_terrain_ruggedness_index(self):
        # Test with a simple 3x3 window
        window = np.array([[1,2,3],[4,5,6],[7,8,9]], dtype=float)
        center_val = 5
        expected_tri = (abs(1-5)+abs(2-5)+abs(3-5)+abs(4-5)+abs(6-5)+abs(7-5)+abs(8-5)+abs(9-5))/8.0
        # (4+3+2+1+1+2+3+4)/8 = 20/8 = 2.5
        assert geo_utils.calculate_terrain_ruggedness_index(window) == expected_tri
        assert np.isnan(geo_utils.calculate_terrain_ruggedness_index(np.array([[1,2],[3,4]]))) # Invalid shape
        assert np.isnan(geo_utils.calculate_terrain_ruggedness_index(np.array([[1,2,3],[4,np.nan,6],[7,8,9]]))) # Center is nan

    def test_calculate_landscape_metrics(self, mock_rasterio_open_with_data, sample_point_a, tmp_path):
        dem_file = tmp_path / "landscape_dem.tif"
        dem_file.touch()
        mock_open, mock_ds = mock_rasterio_open_with_data
        # The mock_ds.read will return a 3x3 array if radius is small enough to make a 3x3 window
        # For this test, we only check TRI. Mean/std will depend on the window_data mock.
        
        metrics = geo_utils.calculate_landscape_metrics(sample_point_a, str(dem_file), radius_m=1) # Small radius
        
        # Expected TRI from the mock 3x3 window data [[10,12,11],[15,20,18],[19,22,21]] center 20
        # (10+8+9+5+2+1+2+1)/8 = 38/8 = 4.75
        assert metrics.get('ruggedness_tri') == 4.75 
        # Check if elevation_mean and stddev are populated (will use the default 10x10 window from mock if radius is large)
        assert 'elevation_mean' in metrics
        assert 'elevation_stddev' in metrics

    def test_determine_geomorphological_context(self, mock_rasterio_open_with_data, sample_point_a, tmp_path):
        dem_file = tmp_path / "geomorph_dem.tif"
        dem_file.touch()
        mock_open, mock_ds = mock_rasterio_open_with_data
        
        # Mock landscape metrics to control context output
        with mock.patch('utils.geo_utils.calculate_landscape_metrics') as mock_calc_metrics:
            # Case 1: Flat plain
            mock_calc_metrics.return_value = {'elevation_stddev': 2.0, 'ruggedness_tri': 1.0}
            context1 = geo_utils.determine_geomorphological_context(sample_point_a, str(dem_file))
            assert context1['landform'] == 'flat_plain'
            assert context1['slope_position'] == 'level'

            # Case 2: Hilly/Mountainous
            mock_calc_metrics.return_value = {'elevation_stddev': 30.0, 'ruggedness_tri': 60.0}
            context2 = geo_utils.determine_geomorphological_context(sample_point_a, str(dem_file))
            assert context2['landform'] == 'hilly_or_mountainous'

class TestDirectionAndRelativePosition:
    @pytest.mark.parametrize("degrees, cardinal", [
        (0, 'N'), (22.5, 'NNE'), (45, 'NE'), (180, 'S'), (270, 'W'), (359, 'N'), (380, 'NNE')
    ])
    def test_direction_to_cardinal(self, degrees, cardinal):
        assert geo_utils.direction_to_cardinal(degrees) == cardinal

    def test_calculate_relative_position(self, sample_point_a, sample_point_b):
        pos = geo_utils.calculate_relative_position(sample_point_a, sample_point_b)
        assert 'distance_m' in pos
        assert 'bearing_degrees' in pos
        assert 'direction' in pos
        # Test that distance is same as haversine
        assert abs(pos['distance_m'] - geo_utils.haversine_distance(sample_point_a.latitude, sample_point_a.longitude, 
                                                                sample_point_b.latitude, sample_point_b.longitude)) < 1e-6 