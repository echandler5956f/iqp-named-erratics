from pathlib import Path
import pytest
import os
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point

from data_pipeline.cache import CacheManager

@pytest.fixture
def cache_manager(tmp_path):
    """Provides a CacheManager instance with a temporary cache directory."""
    cache_dir = tmp_path / "test_cache"
    cache_dir.mkdir()
    return CacheManager(cache_dir=str(cache_dir))

@pytest.fixture
def sample_gdf():
    """Provides a sample GeoDataFrame for testing."""
    data = {
        'id': [1, 2],
        'name': ['Point A', 'Point B'],
        'geometry': [Point(0, 0), Point(1, 1)]
    }
    return gpd.GeoDataFrame(data, crs="EPSG:4326")

@pytest.fixture
def sample_gdf_no_crs():
    """Provides a sample GeoDataFrame without CRS."""
    data = {
        'id': [3],
        'name': ['Point C'],
        'geometry': [Point(2, 2)]
    }
    return gpd.GeoDataFrame(data)

class TestCacheManager:
    def test_save_and_load_gdf(self, cache_manager, sample_gdf):
        source_name = "test_source_1"
        params = {"filter": "A"}

        # Save
        assert cache_manager.save(sample_gdf, source_name, params) == True
        
        cache_file_path = cache_manager._get_cache_path(source_name, params)
        assert os.path.exists(cache_file_path)
        assert os.path.exists(cache_file_path + ".crs")
        assert os.path.exists(cache_file_path + ".geom_col")

        # Load
        loaded_gdf = cache_manager.load(source_name, params)
        assert loaded_gdf is not None
        
        # Basic checks (full equality can be tricky with float precision)
        assert len(loaded_gdf) == len(sample_gdf)
        assert all(loaded_gdf.columns == sample_gdf.columns)
        assert loaded_gdf.crs == sample_gdf.crs
        pd.testing.assert_frame_equal( # Compare non-geometry parts
            loaded_gdf.drop(columns='geometry'), 
            sample_gdf.drop(columns='geometry'),
            check_dtype=False # Feather might alter some dtypes slightly
        )
        assert all(loaded_gdf.geometry.geom_equals(sample_gdf.geometry))

    def test_save_gdf_no_crs(self, cache_manager, sample_gdf_no_crs):
        source_name = "test_source_no_crs"
        assert cache_manager.save(sample_gdf_no_crs, source_name) == True
        loaded_gdf = cache_manager.load(source_name)
        assert loaded_gdf is not None
        assert loaded_gdf.crs is None # Should remain None as per CacheManager logic
        assert len(loaded_gdf) == 1

    def test_is_cached(self, cache_manager, sample_gdf):
        source_name = "test_source_is_cached"
        assert not cache_manager.is_cached(source_name) # Not cached yet
        
        cache_manager.save(sample_gdf, source_name)
        assert cache_manager.is_cached(source_name) # Now cached

        # Test with source_mtime (mocking os.path.getmtime for cache file)
        cache_file_path = cache_manager._get_cache_path(source_name)
        current_time = os.path.getmtime(cache_file_path)
        
        assert cache_manager.is_cached(source_name, source_mtime=current_time - 100) # Cache is newer
        assert not cache_manager.is_cached(source_name, source_mtime=current_time + 100) # Cache is older

    def test_load_non_existent_cache(self, cache_manager):
        assert cache_manager.load("non_existent_source") is None

    def test_clear_specific_source(self, cache_manager, sample_gdf):
        source_name = "test_source_to_clear"
        cache_manager.save(sample_gdf, source_name)
        assert cache_manager.is_cached(source_name)
        
        cache_manager.clear(source_name)
        assert not cache_manager.is_cached(source_name)
        cache_file_path = cache_manager._get_cache_path(source_name)
        assert not os.path.exists(cache_file_path)
        assert not os.path.exists(cache_file_path + ".crs")
        assert not os.path.exists(cache_file_path + ".geom_col")

    def test_clear_all_caches(self, cache_manager, sample_gdf):
        source_name1 = "test_source_clear_all_1"
        source_name2 = "test_source_clear_all_2"
        cache_manager.save(sample_gdf, source_name1)
        cache_manager.save(sample_gdf, source_name2, params={"p": 1})
        
        assert cache_manager.is_cached(source_name1)
        assert cache_manager.is_cached(source_name2, params={"p": 1})
        
        cache_manager.clear() # Clear all
        assert not cache_manager.is_cached(source_name1)
        assert not cache_manager.is_cached(source_name2, params={"p": 1})
        assert len(list(Path(cache_manager.cache_dir).glob("*"))) == 0 # Cache dir should be empty

    def test_cache_paths_differ_with_params(self, cache_manager):
        source_name = "param_diff_source"
        path1 = cache_manager._get_cache_path(source_name, params={"filter": "A"})
        path2 = cache_manager._get_cache_path(source_name, params={"filter": "B"})
        path3 = cache_manager._get_cache_path(source_name, params=None)
        path4 = cache_manager._get_cache_path(source_name) # Same as None
        
        assert path1 != path2
        assert path1 != path3
        assert path2 != path3
        assert path3 == path4

    def test_cache_with_empty_gdf(self, cache_manager):
        source_name = "empty_gdf_source"
        empty_gdf = gpd.GeoDataFrame(columns=['id', 'geometry'], geometry='geometry', crs="EPSG:3857")
        
        assert cache_manager.save(empty_gdf, source_name) == True
        loaded_gdf = cache_manager.load(source_name)
        
        assert loaded_gdf is not None
        assert loaded_gdf.empty
        assert loaded_gdf.crs.to_string() == "EPSG:3857"
        assert 'id' in loaded_gdf.columns
        assert loaded_gdf._geometry_column_name == 'geometry' 