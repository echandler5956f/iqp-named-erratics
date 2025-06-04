#!/usr/bin/env python3
"""
Glacial Erratics Spatial Analysis Runner

This script provides a unified command-line interface for running
various analysis tasks on the North American Named Glacial Erratics dataset.
"""

import os
import sys
import argparse
import subprocess
import logging
from typing import List, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CONDA_ENV_NAME = 'iqp-py310'

def ensure_conda_environment():
    """Check if running in the correct conda environment and try to activate it if not."""
    current_env = os.environ.get('CONDA_DEFAULT_ENV')
    
    if current_env == CONDA_ENV_NAME:
        return True
    
    logger.warning(f"Not running in the expected conda environment '{CONDA_ENV_NAME}'")
    logger.warning(f"Current environment: {current_env or 'Not in a conda environment'}")
    
    try:
        # Try to find conda executable
        conda_executable = None
        if sys.platform == 'win32':
            # Windows-specific logic
            conda_paths = [
                os.path.join(os.environ.get('USERPROFILE', ''), 'miniconda3', 'Scripts', 'conda.exe'),
                os.path.join(os.environ.get('USERPROFILE', ''), 'anaconda3', 'Scripts', 'conda.exe')
            ]
            for path in conda_paths:
                if os.path.exists(path):
                    conda_executable = path
                    break
        else:
            # Unix-like systems
            try:
                process = subprocess.run(['which', 'conda'], capture_output=True, text=True, check=True)
                conda_executable = process.stdout.strip()
            except:
                # Try some standard locations
                conda_paths = [
                    os.path.join(os.environ.get('HOME', ''), 'miniconda3', 'bin', 'conda'),
                    os.path.join(os.environ.get('HOME', ''), 'anaconda3', 'bin', 'conda')
                ]
                for path in conda_paths:
                    if os.path.exists(path):
                        conda_executable = path
                        break
        
        if not conda_executable:
            logger.error("Could not find conda executable")
            return False
        
        # Try to get path to the conda environment
        env_info = subprocess.run(
            [conda_executable, 'env', 'list', '--json'],
            capture_output=True, text=True, check=True
        )
        
        import json
        env_list = json.loads(env_info.stdout)
        env_path = None
        
        for env in env_list.get('envs', []):
            if env.endswith(CONDA_ENV_NAME) or os.path.basename(env) == CONDA_ENV_NAME:
                env_path = env
                break
        
        if not env_path:
            logger.error(f"Could not find environment '{CONDA_ENV_NAME}'")
            logger.error("Please run create_conda_env.sh to set up the environment")
            return False
        
        # If we're here, we know the conda environment exists but isn't active
        logger.warning(f"Please activate the environment with: conda activate {CONDA_ENV_NAME}")
        logger.warning("Then run this script again.")
        
        # On Unix, we can try to provide a wrapped command
        if sys.platform != 'win32':
            cmd_parts = sys.argv.copy()
            cmd = f"conda run -n {CONDA_ENV_NAME} {' '.join(cmd_parts)}"
            logger.info(f"You can also run with: {cmd}")
        
        return False
        
    except Exception as e:
        logger.error(f"Error checking conda environment: {e}")
        return False

def run_proximity_analysis(erratic_id: int, feature_layers: List[str] = None, update_db: bool = False,
                          output_file: str = None, verbose: bool = False):
    """Run proximity analysis for a specific erratic."""
    script_path = os.path.join(SCRIPT_DIR, 'proximity_analysis.py')
    
    cmd = [sys.executable, script_path, str(erratic_id)]
    
    if feature_layers:
        cmd.extend(['--features'] + feature_layers)
    
    if update_db:
        cmd.append('--update-db')
    
    if output_file:
        cmd.extend(['--output', output_file])
    
    if verbose:
        cmd.append('--verbose')
    
    logger.info(f"Running proximity analysis for erratic {erratic_id}")
    logger.debug(f"Command: {' '.join(cmd)}")
    
    try:
        subprocess.run(cmd, check=True)
        logger.info(f"Proximity analysis for erratic {erratic_id} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Proximity analysis failed: {e}")
        return False

def run_classification(erratic_id: int, build_topics: bool = True, update_db: bool = False,
                      output_file: str = None, verbose: bool = False):
    """Run classification for a specific erratic."""
    script_path = os.path.join(SCRIPT_DIR, 'classify_erratic.py')
    
    cmd = [sys.executable, script_path, str(erratic_id)]
    
    if build_topics:
        cmd.append('--build-topics')
    
    if update_db:
        cmd.append('--update-db')
    
    if output_file:
        cmd.extend(['--output', output_file])
    
    if verbose:
        cmd.append('--verbose')
    
    logger.info(f"Running classification for erratic {erratic_id}")
    logger.debug(f"Command: {' '.join(cmd)}")
    
    try:
        subprocess.run(cmd, check=True)
        logger.info(f"Classification for erratic {erratic_id} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Classification failed: {e}")
        return False

def run_clustering(algorithm: str, output_file: str, verbose: bool = False, 
                   k: Optional[int] = None, features: Optional[List[str]] = None, 
                   eps: Optional[float] = None, min_samples: Optional[int] = None, 
                   metric: Optional[str] = None, linkage: Optional[str] = None):
    """Run clustering analysis for all erratics using the specified algorithm and parameters."""
    script_path = os.path.join(SCRIPT_DIR, 'clustering.py')
    
    cmd = [sys.executable, script_path, '--algorithm', algorithm, '--output', output_file]
    
    if k is not None:
        cmd.extend(['--k', str(k)])
    if features:
        cmd.extend(['--features'] + features)
    if eps is not None:
        cmd.extend(['--eps', str(eps)])
    if min_samples is not None:
        cmd.extend(['--min_samples', str(min_samples)])
    if metric:
        cmd.extend(['--metric', metric])
    if linkage:
        cmd.extend(['--linkage', linkage])
    
    if verbose:
        cmd.append('--verbose')
    
    logger.info(f"Running clustering analysis ({algorithm}) for all erratics")
    logger.debug(f"Command: {' '.join(cmd)}")
    
    try:
        subprocess.run(cmd, check=True)
        logger.info("Clustering analysis completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Clustering analysis failed: {e}")
        return False

def run_environment_test(verbose: bool = False):
    """Run environment test to verify all dependencies are installed."""
    script_path = os.path.join(SCRIPT_DIR, 'test_environment.py')
    
    cmd = [sys.executable, script_path]
    
    if verbose:
        cmd.append('--verbose')
    
    logger.info("Running environment test")
    logger.debug(f"Command: {' '.join(cmd)}")
    
    try:
        subprocess.run(cmd, check=True)
        logger.info("Environment test completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Environment test failed: {e}")
        return False

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Glacial Erratics Spatial Analysis Tools')
    subparsers = parser.add_subparsers(dest='command', help='Analysis command to run')
    
    # Proximity analysis parser
    proximity_parser = subparsers.add_parser('proximity', help='Run proximity analysis for an erratic')
    proximity_parser.add_argument('erratic_id', type=int, help='ID of the erratic to analyze')
    proximity_parser.add_argument('--features', nargs='+', help='Feature layers to include')
    proximity_parser.add_argument('--update-db', action='store_true', help='Update database with results')
    proximity_parser.add_argument('--output', type=str, help='Output file for results (JSON)')
    proximity_parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    
    # Classification parser
    classify_parser = subparsers.add_parser('classify', help='Run classification for an erratic')
    classify_parser.add_argument('erratic_id', type=int, help='ID of the erratic to classify')
    classify_parser.add_argument('--no-topics', action='store_true', help='Skip building topic model')
    classify_parser.add_argument('--update-db', action='store_true', help='Update database with results')
    classify_parser.add_argument('--output', type=str, help='Output file for results (JSON)')
    classify_parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    
    # Clustering parser
    cluster_parser = subparsers.add_parser('cluster', help='Run clustering analysis for all erratics')
    cluster_parser.add_argument('--algorithm', type=str, default='dbscan', choices=['dbscan', 'kmeans', 'hierarchical'], help='Clustering algorithm to use (default: dbscan)')
    cluster_parser.add_argument('--output', type=str, required=True, help='Output file for results (JSON)')
    # K-Means / Hierarchical args
    cluster_parser.add_argument('--k', type=int, help='K-Means/Hierarchical: Number of clusters.')
    cluster_parser.add_argument('--features', nargs='+', help='Features for clustering (e.g., longitude latitude, vector_embedding). Default varies by algorithm.')
    cluster_parser.add_argument('--linkage', type=str, choices=['ward', 'complete', 'average', 'single'], help='Hierarchical: Linkage criterion (default: ward).')
    # DBSCAN args
    cluster_parser.add_argument('--eps', type=float, help='DBSCAN: Epsilon parameter. Auto-estimated if not provided.')
    cluster_parser.add_argument('--min_samples', type=int, help='DBSCAN: Minimum number of samples (default: 3).')
    cluster_parser.add_argument('--metric', type=str, choices=['auto', 'haversine', 'euclidean', 'cosine'], help='DBSCAN: Distance metric (default: auto).')
    cluster_parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    
    # Environment test parser
    test_parser = subparsers.add_parser('test-env', help='Test the environment setup')
    test_parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Check if we're in the right conda environment
    if not ensure_conda_environment():
        return 1
    
    # Run the requested command
    if args.command == 'proximity':
        success = run_proximity_analysis(
            args.erratic_id,
            args.features,
            args.update_db,
            args.output,
            args.verbose
        )
    elif args.command == 'classify':
        success = run_classification(
            args.erratic_id,
            not args.no_topics,
            args.update_db,
            args.output,
            args.verbose
        )
    elif args.command == 'cluster':
        success = run_clustering(
            algorithm=args.algorithm,
            output_file=args.output,
            verbose=args.verbose,
            k=args.k,
            features=args.features,
            eps=args.eps,
            min_samples=args.min_samples,
            metric=args.metric,
            linkage=args.linkage
        )
    elif args.command == 'test-env':
        success = run_environment_test(args.verbose)
    else:
        parser.print_help()
        return 0
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main()) 