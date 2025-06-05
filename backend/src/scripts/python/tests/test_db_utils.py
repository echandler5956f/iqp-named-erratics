import pytest
from unittest import mock
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point
import psycopg2 # For mocking errors

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

@pytest.fixture
def mock_conn():
    conn = mock.MagicMock(spec=psycopg2.extensions.connection)
    conn.cursor.return_value = mock.MagicMock(spec=psycopg2.extensions.cursor)
    return conn

@pytest.fixture
def mock_cursor(mock_conn):
    return mock_conn.cursor.return_value

class TestGetDbConnection:
    @mock.patch('psycopg2.connect')
    def test_get_db_connection_success(self, mock_psycopg2_connect, mock_conn):
        mock_psycopg2_connect.return_value = mock_conn
        conn = db_utils.get_db_connection()
        assert conn == mock_conn
        mock_psycopg2_connect.assert_called_once_with(
            host="testhost", database="testdb", user="testuser", password="testpass", port=5432
        )

    def test_get_db_connection_missing_env_var(self, monkeypatch):
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
    def test_load_all_erratics_gdf_success(self, mock_read_postgis, mock_get_conn, mock_conn):
        mock_get_conn.return_value = mock_conn
        sample_data = {
            'id': [1, 2],
            'name': ['Erratic A', 'Erratic B'],
            'location': [Point(1,1), Point(2,2)], # gpd.read_postgis will create geometry
            'vector_embedding': ['[0.1,0.2]', None] # Test string parsing
        }
        mock_gdf = gpd.GeoDataFrame(pd.DataFrame(sample_data), geometry='location', crs="EPSG:4326")
        mock_read_postgis.return_value = mock_gdf

        result_gdf = db_utils.load_all_erratics_gdf()
        
        mock_read_postgis.assert_called_once()
        assert not result_gdf.empty
        assert len(result_gdf) == 2
        assert isinstance(result_gdf.iloc[0]['vector_embedding'], list)
        assert result_gdf.iloc[1]['vector_embedding'] is None
        mock_conn.close.assert_called_once()

    @mock.patch('utils.db_utils.get_db_connection')
    @mock.patch('geopandas.read_postgis', side_effect=Exception("DB Read Error"))
    def test_load_all_erratics_gdf_db_error(self, mock_read_postgis, mock_get_conn, mock_conn):
        mock_get_conn.return_value = mock_conn
        result_gdf = db_utils.load_all_erratics_gdf()
        assert result_gdf.empty
        mock_conn.close.assert_called_once()

class TestLoadErraticDetailsById:
    @mock.patch('utils.db_utils.get_db_connection')
    def test_load_erratic_details_success(self, mock_get_conn, mock_conn, mock_cursor):
        mock_get_conn.return_value = mock_conn
        mock_db_row = {
            'id': 1, 'name': 'Test Erratic', 'longitude': '-70.123', 'latitude': '45.456',
            'elevation': '100.5', 'size_meters': '10.0'
        }
        mock_cursor.fetchone.return_value = mock_db_row
        
        result = db_utils.load_erratic_details_by_id(1)
        
        assert result is not None
        assert result['id'] == 1
        assert result['name'] == 'Test Erratic'
        assert result['longitude'] == -70.123
        assert result['latitude'] == 45.456
        assert result['elevation'] == 100.5
        assert result['size_meters'] == 10.0
        mock_conn.close.assert_called_once()

    @mock.patch('utils.db_utils.get_db_connection')
    def test_load_erratic_details_not_found(self, mock_get_conn, mock_conn, mock_cursor):
        mock_get_conn.return_value = mock_conn
        mock_cursor.fetchone.return_value = None
        result = db_utils.load_erratic_details_by_id(999)
        assert result is None
        mock_conn.close.assert_called_once()

class TestUpdateErraticAnalysisResults:
    @mock.patch('utils.db_utils.get_db_connection')
    def test_update_insert_path(self, mock_get_conn, mock_conn, mock_cursor):
        mock_get_conn.return_value = mock_conn
        mock_cursor.fetchone.return_value = None # Simulate record does not exist (insert path)
        
        erratic_id = 10
        analysis_data = {
            'usage_type': ['landmark'], 
            'cultural_significance_score': 7,
            'nearest_natd_road_dist': 123.45,
            'vector_embedding': [0.1, 0.2, 0.3] # Example embedding
        }
        
        success = db_utils.update_erratic_analysis_results(erratic_id, analysis_data)
        
        assert success == True
        mock_cursor.execute.assert_any_call(
            'SELECT "erraticId" FROM "ErraticAnalyses" WHERE "erraticId" = %s;', (erratic_id,)
        )
        # Check the INSERT call (actual SQL might be complex to match exactly, check key parts)
        insert_call_args = mock_cursor.execute.call_args_list[1][0] # Second execute call is INSERT
        assert "INSERT INTO \"ErraticAnalyses\"" in insert_call_args[0]
        assert "usage_type" in insert_call_args[0]
        assert "nearest_natd_road_dist" in insert_call_args[0]
        assert "vector_embedding" in insert_call_args[0]
        assert erratic_id in insert_call_args[1] # erraticId should be in values
        mock_conn.commit.assert_called_once()
        mock_conn.close.assert_called_once()

    @mock.patch('utils.db_utils.get_db_connection')
    def test_update_update_path(self, mock_get_conn, mock_conn, mock_cursor):
        mock_get_conn.return_value = mock_conn
        mock_cursor.fetchone.return_value = {'erraticId': 20} # Simulate record exists (update path)

        erratic_id = 20
        analysis_data = {
            'accessibility_score': 3,
            'nearest_forest_trail_dist': 50.0
        }
        success = db_utils.update_erratic_analysis_results(erratic_id, analysis_data)

        assert success == True
        # Check the UPDATE call
        update_call_args = mock_cursor.execute.call_args_list[1][0]
        assert "UPDATE \"ErraticAnalyses\" SET" in update_call_args[0]
        assert "accessibility_score" in update_call_args[0]
        assert "nearest_forest_trail_dist" in update_call_args[0]
        assert "WHERE \"erraticId\" = %s" in update_call_args[0]
        assert erratic_id in update_call_args[1]
        mock_conn.commit.assert_called_once()
        mock_conn.close.assert_called_once()

    @mock.patch('utils.db_utils.get_db_connection')
    def test_update_no_valid_data(self, mock_get_conn, mock_conn, mock_cursor):
        mock_get_conn.return_value = mock_conn
        success = db_utils.update_erratic_analysis_results(30, {'invalid_field': 'value'})
        assert success == False
        mock_cursor.execute.assert_not_called() # Should not try to execute if no valid fields
        mock_conn.commit.assert_not_called()
        mock_conn.close.assert_called_once() # Connection still opened and closed

    @mock.patch('utils.db_utils.get_db_connection')
    def test_update_db_error(self, mock_get_conn, mock_conn, mock_cursor):
        mock_get_conn.return_value = mock_conn
        mock_cursor.fetchone.return_value = None # insert path
        mock_cursor.execute.side_effect = [None, psycopg2.Error("DB Error on execute")] # First for select, second for insert/update

        erratic_id = 40
        analysis_data = {'nearest_natd_road_dist': 100.0}
        success = db_utils.update_erratic_analysis_results(erratic_id, analysis_data)

        assert success == False
        mock_conn.rollback.assert_called_once()
        mock_conn.close.assert_called_once() 