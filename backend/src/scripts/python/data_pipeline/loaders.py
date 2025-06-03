"""
Data loaders for different protocols.

Each loader is responsible for acquiring data from a specific type of source
(HTTP, FTP, local file, database, etc.) and saving it locally.
"""

import os
import requests
import ftplib
import tempfile
import logging
from typing import Optional
from urllib.parse import urlparse
from abc import ABC, abstractmethod

from .sources import DataSource


logger = logging.getLogger(__name__)


class BaseLoader(ABC):
    """Abstract base class for data loaders"""
    
    @abstractmethod
    def load(self, source: DataSource, target_path: str) -> bool:
        """Load data from source and save to target_path"""
        pass


class HTTPLoader(BaseLoader):
    """Loader for HTTP/HTTPS sources"""
    
    def load(self, source: DataSource, target_path: str) -> bool:
        """Download file via HTTP/HTTPS"""
        if not source.url:
            logger.error(f"No URL provided for source {source.name}")
            return False
            
        try:
            logger.info(f"Downloading {source.name} from {source.url}")
            
            # Set headers to avoid 403 errors from APIs
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            
            response = requests.get(source.url, stream=True, headers=headers)
            response.raise_for_status()
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(target_path), exist_ok=True)
            
            # Download to temporary file first
            with tempfile.NamedTemporaryFile(delete=False, dir=os.path.dirname(target_path)) as tmp_file:
                for chunk in response.iter_content(chunk_size=8192):
                    tmp_file.write(chunk)
                tmp_path = tmp_file.name
            
            # Move to final location
            os.rename(tmp_path, target_path)
            logger.info(f"Successfully downloaded {source.name} to {target_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error downloading {source.name}: {e}")
            return False


class FTPLoader(BaseLoader):
    """Loader for FTP sources"""
    
    def load(self, source: DataSource, target_path: str) -> bool:
        """Download file via FTP"""
        if not source.url:
            logger.error(f"No URL provided for source {source.name}")
            return False
        
        try:
            parsed = urlparse(source.url)
            host = parsed.netloc
            path = parsed.path
            filename = os.path.basename(path) or source.params.get('filename', 'data')
            
            logger.info(f"Connecting to FTP server {host}")
            with ftplib.FTP(host) as ftp:
                ftp.login()  # Anonymous login
                
                if path and path != '/':
                    ftp.cwd(os.path.dirname(path))
                
                # Ensure directory exists
                os.makedirs(os.path.dirname(target_path), exist_ok=True)
                
                # Download file
                with open(target_path, 'wb') as f:
                    logger.info(f"Downloading {filename} from FTP")
                    ftp.retrbinary(f'RETR {filename}', f.write)
                    
            logger.info(f"Successfully downloaded {source.name} to {target_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error downloading {source.name} via FTP: {e}")
            return False


class FileLoader(BaseLoader):
    """Loader for local files"""
    
    def load(self, source: DataSource, target_path: str) -> bool:
        """Copy or link local file"""
        if not source.path:
            logger.error(f"No path provided for source {source.name}")
            return False
            
        if not os.path.exists(source.path):
            logger.error(f"Source file not found: {source.path}")
            return False
        
        # For local files, we can just return the path
        # No need to copy unless specifically requested
        if source.params.get('copy', False):
            import shutil
            os.makedirs(os.path.dirname(target_path), exist_ok=True)
            shutil.copy2(source.path, target_path)
            logger.info(f"Copied {source.path} to {target_path}")
        else:
            # Just symlink or return the original path
            logger.info(f"Using local file {source.path} directly")
            
        return True


class LoaderFactory:
    """Factory for creating appropriate loaders"""
    
    _loaders = {
        'http': HTTPLoader,
        'https': HTTPLoader,
        'ftp': FTPLoader,
        'file': FileLoader,
    }
    
    @classmethod
    def get_loader(cls, source_type: str) -> BaseLoader:
        """Get appropriate loader for source type"""
        loader_class = cls._loaders.get(source_type.lower())
        if not loader_class:
            raise ValueError(f"No loader available for source type: {source_type}")
        return loader_class() 