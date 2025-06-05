import pytest
from unittest import mock
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point
import psycopg2 # For mocking errors
from psycopg2.extras import RealDictCursor # Import for type checking if needed by RealDictCursor spec

# Module to test
from utils import db_utils

# Mock environment variables for database connection for all tests in this module
@pytest.fixture(autouse=True)
def mock_db_env_vars(monkeypatch):
    monkeypatch.setenv("DB_HOST", "testhost")
    monkeypatch.setenv("DB_NAME", "testdb")
    monkeypatch.setenv("DB_USER", "testuser")
    monkeypatch.setenv("DB_PASSWORD", "testpass")
    monkeypatch.setenv("DB_PORT", "5432")

# This mock_cursor will be the one directly configured by tests AND used by the code.
@pytest.fixture
def mock_cursor_fixture(): 
    # Use a simple MagicMock without strict spec to ensure execute call tracking
    cursor = mock.MagicMock()
    cursor.execute = mock.MagicMock()
    cursor.fetchone = mock.MagicMock()
    cursor.__enter__.return_value = cursor
    cursor.__exit__.return_value = False
    return cursor

@pytest.fixture
def mock_conn_fixture(mock_cursor_fixture):
    conn = mock.MagicMock(spec=psycopg2.extensions.connection)
    # Ensure that any call to conn.cursor() returns our pre-configured mock_cursor.
    conn.cursor.return_value = mock_cursor_fixture
    return conn

# The mock_cursor fixture now simply passes through the pre-configured one.
# Tests will inject 'mock_cursor' and it will be the 'configured_mock_cursor'.
@pytest.fixture
def mock_cursor(mock_cursor_fixture):
    return mock_cursor_fixture

class TestGetDbConnection:
    @mock.patch('psycopg2.connect')
    def test_get_db_connection_success(self, mock_psycopg2_connect):
        # Simplified: get_db_connection just needs to return a mock connection here.
        # The detailed cursor mocking is for other functions.
        mock_connection_instance = mock.MagicMock(spec=psycopg2.extensions.connection)
        mock_psycopg2_connect.return_value = mock_connection_instance
        conn = db_utils.get_db_connection()
        assert conn == mock_connection_instance
        mock_psycopg2_connect.assert_called_once_with(
            host="testhost", database="testdb", user="testuser", password="testpass", port=5432
        )

    @mock.patch('utils.db_utils.load_dotenv')  # Mock load_dotenv to prevent reloading from .env file
    def test_get_db_connection_missing_env_var(self, mock_load_dotenv, monkeypatch):
        monkeypatch.delenv("DB_NAME") # Remove a required var
        with pytest.raises(ValueError, match="DB config incomplete. Missing:.*database"):
            db_utils.get_db_connection()

    @mock.patch('psycopg2.connect', side_effect=psycopg2.OperationalError("Connection failed"))
    def test_get_db_connection_psycopg2_error(self, mock_psycopg2_connect):
        with pytest.raises(psycopg2.OperationalError, match="Connection failed"):
            db_utils.get_db_connection()

class TestLoadAllErraticsGdf:
    @mock.patch('utils.db_utils.get_db_connection')
    @mock.patch('geopandas.read_postgis')
    def test_load_all_erratics_gdf_success(self, mock_read_postgis, mock_get_conn, mock_conn_fixture):
        mock_get_conn.return_value = mock_conn_fixture
        sample_data = {
            'id': [1, 2],
            'name': ['Erratic A', 'Erratic B'],
            'location': [Point(1,1), Point(2,2)], # gpd.read_postgis will create geometry
            'vector_embedding': ['[0.1,0.2]', None] # Test string parsing
        }
        mock_gdf = gpd.GeoDataFrame(pd.DataFrame(sample_data), geometry='location', crs="EPSG:4326")
        mock_read_postgis.return_value = mock_gdf

        result_gdf = db_utils.load_all_erratics_gdf()
        
        mock_read_postgis.assert_called_once_with(mock.ANY, mock_conn_fixture, geom_col='location', crs='EPSG:4326')
        assert not result_gdf.empty
        assert len(result_gdf) == 2
        assert isinstance(result_gdf.iloc[0]['vector_embedding'], list)
        assert result_gdf.iloc[1]['vector_embedding'] is None
        mock_conn_fixture.close.assert_called_once()

    @mock.patch('utils.db_utils.get_db_connection')
    @mock.patch('geopandas.read_postgis', side_effect=Exception("DB Read Error"))
    def test_load_all_erratics_gdf_db_error(self, mock_read_postgis, mock_get_conn, mock_conn_fixture):
        mock_get_conn.return_value = mock_conn_fixture
        result_gdf = db_utils.load_all_erratics_gdf()
        assert result_gdf.empty
        mock_conn_fixture.close.assert_called_once()

class TestLoadErraticDetailsById:
    @mock.patch('utils.db_utils.get_db_connection')
    def test_load_erratic_details_success(self, mock_get_conn, mock_conn_fixture, mock_cursor_fixture):
        # mock_get_conn is patched to return mock_conn.
        # mock_conn.cursor() will return mock_cursor (which is configured_mock_cursor).
        mock_get_conn.return_value = mock_conn_fixture
        
        mock_db_row = {
            'id': 1, 'name': 'Test Erratic', 'longitude': '-70.123', 'latitude': '45.456',
            'elevation': '100.5', 'size_meters': '10.0'
        }
        mock_cursor_fixture.fetchone.return_value = mock_db_row
        
        result = db_utils.load_erratic_details_by_id(1)
        
        assert result is not None
        assert result['id'] == 1
        assert result['name'] == 'Test Erratic'
        assert result['longitude'] == -70.123 # Code converts to float
        assert result['latitude'] == 45.456  # Code converts to float
        assert result['elevation'] == 100.5   # Code converts to float
        assert result['size_meters'] == 10.0  # Code converts to float
        mock_conn_fixture.close.assert_called_once()
        # Ensure cursor was obtained correctly if needed for debugging
        mock_conn_fixture.cursor.assert_called_once_with(cursor_factory=RealDictCursor)

    @mock.patch('utils.db_utils.get_db_connection')
    def test_load_erratic_details_not_found(self, mock_get_conn, mock_conn_fixture, mock_cursor_fixture):
        mock_get_conn.return_value = mock_conn_fixture
        mock_cursor_fixture.fetchone.return_value = None
        
        result = db_utils.load_erratic_details_by_id(999)
        
        assert result is None
        mock_conn_fixture.close.assert_called_once()
        mock_conn_fixture.cursor.assert_called_once_with(cursor_factory=RealDictCursor)

class TestUpdateErraticAnalysisResults:
    @mock.patch('utils.db_utils.get_db_connection')
    def test_update_insert_path(self, mock_get_conn, mock_conn_fixture, mock_cursor_fixture):
        mock_get_conn.return_value = mock_conn_fixture
        mock_cursor_fixture.fetchone.return_value = None # Record does not exist
        
        # We expect two calls to execute: SELECT then INSERT
        # To handle different behaviors for sequence of calls, use side_effect on execute
        mock_cursor_fixture.execute.side_effect = [
            None, # For the SELECT call
            None  # For the INSERT call
        ]

        erratic_id = 10
        analysis_data = {
            'usage_type': ['landmark'], 
            'cultural_significance_score': 7,
            'nearest_natd_road_dist': 123.45,
            'vector_embedding': [0.1, 0.2, 0.3] 
        }
        
        success = db_utils.update_erratic_analysis_results(erratic_id, analysis_data)
        
        assert success == True
        assert mock_cursor_fixture.execute.call_count == 2
        mock_cursor_fixture.execute.assert_any_call(
            'SELECT "erraticId" FROM "ErraticAnalyses" WHERE "erraticId" = %s;', (erratic_id,)
        )
        # Check the INSERT call (second call)
        insert_call_args = mock_cursor_fixture.execute.call_args_list[1][0]
        assert "INSERT INTO \"ErraticAnalyses\"" in insert_call_args[0]
        assert "usage_type" in insert_call_args[0]
        assert "nearest_natd_road_dist" in insert_call_args[0]
        assert "vector_embedding" in insert_call_args[0]
        assert erratic_id in insert_call_args[1] 
        mock_conn_fixture.commit.assert_called_once()
        mock_conn_fixture.close.assert_called_once()
        mock_conn_fixture.cursor.assert_called_once_with(cursor_factory=mock.ANY) # Allow any factory or none

    @mock.patch('utils.db_utils.get_db_connection')
    def test_update_update_path(self, mock_get_conn, mock_conn_fixture, mock_cursor_fixture):
        mock_get_conn.return_value = mock_conn_fixture
        mock_cursor_fixture.fetchone.return_value = {'erraticId': 20} # Record exists

        # Expect two calls to execute: SELECT then UPDATE
        mock_cursor_fixture.execute.side_effect = [
            None, # For the SELECT call
            None  # For the UPDATE call
        ]

        erratic_id = 20
        analysis_data = {
            'accessibility_score': 3,
            'nearest_forest_trail_dist': 50.0
        }
        success = db_utils.update_erratic_analysis_results(erratic_id, analysis_data)

        assert success == True
        assert mock_cursor_fixture.execute.call_count == 2
        mock_cursor_fixture.execute.assert_any_call(
            'SELECT "erraticId" FROM "ErraticAnalyses" WHERE "erraticId" = %s;', (erratic_id,)
        )
        update_call_args = mock_cursor_fixture.execute.call_args_list[1][0]
        assert "UPDATE \"ErraticAnalyses\" SET" in update_call_args[0]
        assert "accessibility_score" in update_call_args[0]
        assert "nearest_forest_trail_dist" in update_call_args[0]
        assert "WHERE \"erraticId\" = %s" in update_call_args[0]
        assert erratic_id in update_call_args[1]
        mock_conn_fixture.commit.assert_called_once()
        mock_conn_fixture.close.assert_called_once()
        mock_conn_fixture.cursor.assert_called_once_with(cursor_factory=mock.ANY)

    @mock.patch('utils.db_utils.get_db_connection')
    def test_update_no_valid_data(self, mock_get_conn, mock_conn_fixture, mock_cursor_fixture):
        # get_db_connection is patched, but might not be called if logic exits early.
        # mock_conn and mock_cursor are passed but might not be used if get_db_connection().cursor() isn't reached.

        success = db_utils.update_erratic_analysis_results(30, {'invalid_field': 'value'})
        assert success == False
        
        # Assert that get_db_connection was NOT called because the function should exit early
        mock_get_conn.assert_not_called()
        # Consequently, cursor, commit, and close on the mock_conn should not be called either
        mock_cursor_fixture.execute.assert_not_called()
        mock_conn_fixture.commit.assert_not_called()
        mock_conn_fixture.close.assert_not_called() 

    @mock.patch('utils.db_utils.get_db_connection')
    def test_update_db_error(self, mock_get_conn, mock_conn_fixture, mock_cursor_fixture):
        mock_get_conn.return_value = mock_conn_fixture
        mock_cursor_fixture.fetchone.return_value = None # Insert path
        
        # First execute (SELECT) is fine, second (INSERT) raises error
        mock_cursor_fixture.execute.side_effect = [
            None, # For the SELECT call that precedes fetchone. This call itself doesn't return via execute.
            psycopg2.Error("DB Error on execute for INSERT") 
        ]

        erratic_id = 40
        analysis_data = {'nearest_natd_road_dist': 100.0}
        success = db_utils.update_erratic_analysis_results(erratic_id, analysis_data)

        assert success == False
        assert mock_cursor_fixture.execute.call_count == 2 # SELECT then failing INSERT
        mock_conn_fixture.rollback.assert_called_once()
        mock_conn_fixture.close.assert_called_once()
        mock_conn_fixture.cursor.assert_called_once_with(cursor_factory=mock.ANY) 