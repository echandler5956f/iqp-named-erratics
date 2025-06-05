import pytest
import os
from unittest import mock

from data_pipeline.sources import DataSource
from data_pipeline.loaders import FileLoader, LoaderFactory, HTTPLoader, FTPLoader, ManualLoader


@pytest.fixture
def mock_source_file(tmp_path):
    """Create a dummy source file for testing FileLoader."""
    source_file = tmp_path / "source_data.txt"
    source_file.write_text("dummy content")
    return str(source_file)

@pytest.fixture
def target_dir(tmp_path):
    """Create a dummy target directory."""
    target = tmp_path / "target"
    target.mkdir()
    return str(target)

class TestFileLoader:
    def test_load_existing_file_no_copy(self, mock_source_file, target_dir):
        """Test FileLoader.load when file exists and copy is False (default)."""
        source_ds = DataSource(name="test_file_source", source_type="file", format="txt", path=mock_source_file)
        loader = FileLoader()
        target_file_path = os.path.join(target_dir, "test_data.txt")

        with mock.patch('shutil.copy2') as mock_copy:
            assert loader.load(source_ds, target_file_path) == True
            mock_copy.assert_not_called() # Should not copy by default

    def test_load_existing_file_with_copy(self, mock_source_file, target_dir):
        """Test FileLoader.load when file exists and copy is True."""
        source_ds = DataSource(name="test_file_source_copy", source_type="file", format="txt", 
                               path=mock_source_file, params={'copy': True})
        loader = FileLoader()
        target_file_path = os.path.join(target_dir, "test_data_copy.txt")

        with mock.patch('shutil.copy2') as mock_copy:
            assert loader.load(source_ds, target_file_path) == True
            mock_copy.assert_called_once_with(mock_source_file, target_file_path)
            # Check if target directory was created for the copy
            assert os.path.exists(os.path.dirname(target_file_path))

    def test_load_no_source_path(self, target_dir):
        """Test FileLoader.load when source.path is None."""
        source_ds = DataSource(name="test_no_path", source_type="file", format="txt", path=None)
        loader = FileLoader()
        target_file_path = os.path.join(target_dir, "test_data.txt")
        assert loader.load(source_ds, target_file_path) == False

    def test_load_source_file_not_exists(self, target_dir):
        """Test FileLoader.load when source.path does not exist."""
        non_existent_path = "/tmp/non_existent_dummy_file.txt"
        source_ds = DataSource(name="test_not_exists", source_type="file", format="txt", path=non_existent_path)
        loader = FileLoader()
        target_file_path = os.path.join(target_dir, "test_data.txt")
        assert loader.load(source_ds, target_file_path) == False

class TestLoaderFactory:
    def test_get_file_loader(self):
        loader = LoaderFactory.get_loader('file')
        assert isinstance(loader, FileLoader)

    def test_get_http_loader(self):
        loader = LoaderFactory.get_loader('http')
        assert isinstance(loader, HTTPLoader)
        loader_https = LoaderFactory.get_loader('https')
        assert isinstance(loader_https, HTTPLoader)

    def test_get_ftp_loader(self):
        loader = LoaderFactory.get_loader('ftp')
        assert isinstance(loader, FTPLoader)
    
    def test_get_manual_loader(self):
        loader = LoaderFactory.get_loader('manual')
        assert isinstance(loader, ManualLoader)

    def test_get_unknown_loader(self):
        with pytest.raises(ValueError, match="No loader available for source type: unknown"):
            LoaderFactory.get_loader('unknown') 