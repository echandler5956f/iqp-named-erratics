import pytest
from unittest import mock
import geopandas as gpd
from shapely.geometry import Point

# Module to test
from utils import geo_utils
from proximity_analysis import calculate_proximity, main as proximity_main # renamed main to avoid conflict

# --- Fixtures ---
@pytest.fixture
def mock_erratic_data_valid():
    return {
        'id': 1,
        'name': 'Test Erratic Alpha',
        'longitude': -75.5,
        'latitude': 45.5,
        'elevation': 100.0 # Initial DB elevation
    }

@pytest.fixture
def mock_erratic_data_no_loc():
    return {
        'id': 2,
        'name': 'Test Erratic Beta',
        'longitude': None,
        'latitude': None,
        'elevation': 50.0
    }

# Mock GeoDataFrames for various data sources
@pytest.fixture
def gdf_lakes():
    return gpd.GeoDataFrame({
        'id': [101], 'Lake_name': ['Mock Lake'], 'geometry': [Point(-75.501, 45.501)]
    }, crs="EPSG:4326")

@pytest.fixture
def gdf_rivers():
    return gpd.GeoDataFrame({
        'id': [201], 'HYR_NAME': ['Mock River'], 'geometry': [Point(-75.499, 45.499)]
    }, crs="EPSG:4326")

@pytest.fixture
def gdf_territories():
    return gpd.GeoDataFrame({
        'id': [301], 'Name': ['Mock Territory'], 'geometry': [Point(-75.502, 45.502)]
    }, crs="EPSG:4326")

@pytest.fixture
def gdf_settlements(): # Mock for 'osm_north_america' source
    return gpd.GeoDataFrame({
        'id': [401], 'name': ['Mockville'], 'place': ['village'], 'geometry': [Point(-75.498, 45.498)]
    }, crs="EPSG:4326")

@pytest.fixture
def gdf_natd_roads():
    return gpd.GeoDataFrame({
        'id': [501], 'FULLNAME': ['Mock Road'], 'RTTYP': ['I'], 'geometry': [Point(-75.503, 45.503)] 
    }, crs="EPSG:4326")

@pytest.fixture
def gdf_forest_trails():
    return gpd.GeoDataFrame({
        'id': [601], 'TRAIL_NAME': ['Mock Trail'], 'geometry': [Point(-75.497, 45.497)]
    }, crs="EPSG:4326")

# --- Mocks for Patches ---
@pytest.fixture
def mock_db_load_details(mock_erratic_data_valid):
    with mock.patch('utils.db_utils.load_erratic_details_by_id') as mock_load:
        mock_load.return_value = mock_erratic_data_valid
        yield mock_load

@pytest.fixture
def mock_pipeline_load_data(gdf_lakes, gdf_rivers, gdf_territories, gdf_settlements, gdf_natd_roads, gdf_forest_trails):
    def side_effect_load_data(source_name, **kwargs):
        if source_name == 'hydrosheds_lakes': return gdf_lakes
        if source_name == 'hydrosheds_rivers': return gdf_rivers
        if source_name == 'native_territories': return gdf_territories
        if source_name == 'osm_north_america': return gdf_settlements # Used for settlements
        if source_name == 'natd_roads': return gdf_natd_roads
        if source_name == 'forest_trails': return gdf_forest_trails
        return gpd.GeoDataFrame() # Default empty for unknown or unmocked
    
    with mock.patch('proximity_analysis.load_data') as mock_load:
        mock_load.side_effect = side_effect_load_data
        yield mock_load

@pytest.fixture
def mock_geo_utils_dem(tmp_path):
    dummy_dem_path = tmp_path / "dummy_dem_for_proximity.tif"
    dummy_dem_path.touch()
    with mock.patch('utils.geo_utils.load_dem_data') as mock_load_dem:
        mock_load_dem.return_value = str(dummy_dem_path)
        yield mock_load_dem

@pytest.fixture
def mock_geo_utils_raster_ops(monkeypatch): # Mocks functions that use rasterio
    mock_get_elev = mock.MagicMock(return_value=150.0) # Mocked DEM elevation
    mock_calc_landscape = mock.MagicMock(return_value={'ruggedness_tri': 5.5})
    mock_det_geomorph = mock.MagicMock(return_value={'landform': 'test_landform', 'slope_position': 'test_slope_pos'})
    
    monkeypatch.setattr('utils.geo_utils.get_elevation_at_point', mock_get_elev)
    monkeypatch.setattr('utils.geo_utils.calculate_landscape_metrics', mock_calc_landscape)
    monkeypatch.setattr('utils.geo_utils.determine_geomorphological_context', mock_det_geomorph)
    return mock_get_elev, mock_calc_landscape, mock_det_geomorph

# --- Test Class for calculate_proximity ---
class TestCalculateProximity:
    def test_successful_run(self, mock_db_load_details, mock_pipeline_load_data, 
                            mock_geo_utils_dem, mock_geo_utils_raster_ops, mock_erratic_data_valid):
        erratic_id = mock_erratic_data_valid['id']
        results = calculate_proximity(erratic_id)

        assert 'error' not in results
        assert results['erratic_id'] == erratic_id
        assert results['erratic_name'] == mock_erratic_data_valid['name']
        
        pa = results['proximity_analysis']
        assert pa['in_north_america'] == True # Based on mock_erratic_data_valid coords
        
        # Check a few key proximity results (exact distances depend on mock data & haversine)
        assert 'nearest_water_body_dist' in pa
        assert pa['nearest_water_body_type'] == 'lake' # lake is closer in mock GDFs based on coordinates
        assert 'nearest_native_territory_dist' in pa
        assert 'nearest_settlement_dist' in pa
        assert 'nearest_natd_road_dist' in pa
        assert pa['nearest_road_dist'] == pa['nearest_natd_road_dist'] # Check if it got populated
        assert 'nearest_forest_trail_dist' in pa
        
        # Check DEM derived fields from mocks
        assert pa['elevation_dem'] == 150.0
        assert pa['elevation_category'] == 'lowland' # 150m is lowland
        assert pa['ruggedness_tri'] == 5.5
        assert pa['terrain_landform'] == 'test_landform'
        assert pa['terrain_slope_position'] == 'test_slope_pos'
        assert 'accessibility_score' in pa # Should be calculated
        assert 'estimated_displacement_dist' in pa # Should be calculated

    def test_erratic_not_found(self, mock_db_load_details):
        mock_db_load_details.return_value = None
        results = calculate_proximity(999)
        assert 'error' in results
        assert results['error'] == "Erratic with ID 999 not found"

    def test_erratic_missing_location(self, mock_db_load_details, mock_erratic_data_no_loc):
        mock_db_load_details.return_value = mock_erratic_data_no_loc
        results = calculate_proximity(mock_erratic_data_no_loc['id'])
        assert 'error' in results
        assert results['error'] == "Missing location data for erratic"

    def test_load_data_returns_none_for_source(self, mock_db_load_details, mock_pipeline_load_data,
                                               mock_geo_utils_dem, mock_geo_utils_raster_ops, mock_erratic_data_valid,
                                               gdf_rivers, gdf_territories, gdf_settlements, gdf_natd_roads, gdf_forest_trails):
        # Make one of the load_data calls return None (or empty GDF)
        def selective_none_load(source_name, **kwargs):
            if source_name == 'hydrosheds_lakes': return None
            # Use the original side_effect for others
            if source_name == 'hydrosheds_rivers': return gdf_rivers
            if source_name == 'native_territories': return gdf_territories
            if source_name == 'osm_north_america': return gdf_settlements
            if source_name == 'natd_roads': return gdf_natd_roads
            if source_name == 'forest_trails': return gdf_forest_trails
            return gpd.GeoDataFrame()
        
        mock_pipeline_load_data.side_effect = selective_none_load
        
        results = calculate_proximity(mock_erratic_data_valid['id'])
        pa = results['proximity_analysis']
        # If lakes are None, and rivers are present, river data should still be used for water body
        # Depending on the logic this might mean nearest_water_body_dist is based on rivers or None
        # In our mock setup, rivers are closer, so this should still populate based on rivers.
        # If both were None, then the fields would be None.
        assert 'nearest_water_body_dist' in pa 
        # To specifically test one source failing to load and its field being None:
        # e.g. if natd_roads fails, nearest_natd_road_dist should be None
        mock_pipeline_load_data.side_effect = lambda s, **k: None if s == 'natd_roads' else gpd.GeoDataFrame() 
        results_no_roads = calculate_proximity(mock_erratic_data_valid['id'])
        assert results_no_roads['proximity_analysis'].get('nearest_natd_road_dist') is None
        assert results_no_roads['proximity_analysis'].get('nearest_road_dist') is None

    def test_load_dem_data_returns_none(self, mock_db_load_details, mock_pipeline_load_data, 
                                      mock_geo_utils_dem, mock_geo_utils_raster_ops, mock_erratic_data_valid):
        mock_geo_utils_dem.return_value = None # Simulate DEM not found
        results = calculate_proximity(mock_erratic_data_valid['id'])
        pa = results['proximity_analysis']
        assert pa.get('elevation_dem') is None
        assert pa.get('ruggedness_tri') is None 
        assert pa.get('terrain_landform') is None
        assert pa.get('terrain_slope_position') is None
        # elevation_category should fall back to DB elevation if DEM fails
        assert pa['elevation_category'] == geo_utils.get_elevation_category(mock_erratic_data_valid['elevation'])

# --- Test Class for main CLI ---
@mock.patch('proximity_analysis.calculate_proximity')
@mock.patch('utils.db_utils.update_erratic_analysis_results')
@mock.patch('utils.file_utils.json_to_file')
class TestProximityMain:
    def test_main_success_no_db_no_output(self, mock_json_to_file, mock_update_db, mock_calc_prox, mock_erratic_data_valid):
        mock_args = mock.MagicMock()
        mock_args.erratic_id = 1
        mock_args.update_db = False
        mock_args.output = None
        mock_args.verbose = False
        
        mock_calc_prox.return_value = {
            "erratic_id": 1, "proximity_analysis": {"some_key": "some_value"}
        }
        with mock.patch('argparse.ArgumentParser.parse_args', return_value=mock_args):
            with mock.patch('sys.exit') as mock_sys_exit:
                proximity_main()
                mock_sys_exit.assert_called_once_with(None) # Should exit cleanly
        
        mock_calc_prox.assert_called_once_with(1)
        mock_update_db.assert_not_called()
        mock_json_to_file.assert_not_called()

    def test_main_success_with_db_and_output(self, mock_json_to_file, mock_update_db, mock_calc_prox, mock_erratic_data_valid):
        mock_args = mock.MagicMock()
        mock_args.erratic_id = 1
        mock_args.update_db = True
        mock_args.output = "output.json"
        mock_args.verbose = True
        
        calc_results_payload = {"key": "value", "dist": 123.0}
        mock_calc_prox.return_value = {"erratic_id": 1, "proximity_analysis": calc_results_payload}
        mock_update_db.return_value = True

        with mock.patch('argparse.ArgumentParser.parse_args', return_value=mock_args):
            with mock.patch('sys.exit') as mock_sys_exit:
                 with mock.patch('logging.getLogger') as mock_get_logger: # For verbose
                    proximity_main()
                    mock_sys_exit.assert_called_once_with(None)
        
        mock_calc_prox.assert_called_once_with(1)
        mock_update_db.assert_called_once_with(1, calc_results_payload)
        mock_json_to_file.assert_called_once_with(mock_calc_prox.return_value, "output.json")

    def test_main_calc_prox_error(self, mock_json_to_file, mock_update_db, mock_calc_prox):
        mock_args = mock.MagicMock()
        mock_args.erratic_id = 1
        mock_args.update_db = False
        mock_args.output = None
        mock_args.verbose = False
        
        mock_calc_prox.return_value = {"error": "Calculation failed", "erratic_id": 1}
        
        with mock.patch('argparse.ArgumentParser.parse_args', return_value=mock_args):
            with mock.patch('sys.exit') as mock_sys_exit:
                proximity_main()
                mock_sys_exit.assert_called_once_with(1) # Should exit with error code
        
        mock_calc_prox.assert_called_once_with(1)
        mock_update_db.assert_not_called()
        mock_json_to_file.assert_not_called() 