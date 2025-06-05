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

# Ensure scripts directory's parent (python/) is on path for utils imports used in pipeline mode
PYTHON_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
if PYTHON_SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, PYTHON_SCRIPTS_DIR)

def get_results_directory() -> str:
    """Get the default results directory path and ensure it exists.
    
    Returns:
        Path to the data/results directory relative to this script.
    """
    results_dir = os.path.join(SCRIPT_DIR, 'data', 'results')
    os.makedirs(results_dir, exist_ok=True)
    return results_dir

def get_default_output_path(filename: str) -> str:
    """Get a default output file path in the results directory.
    
    Args:
        filename: The base filename for the output file.
        
    Returns:
        Full path to the output file in the results directory.
    """
    return os.path.join(get_results_directory(), filename)

def run_proximity_analysis(
    erratic_id: int,
    update_db: bool = False,
    output_file: Optional[str] = None,
    verbose: bool = False,
):
    """Wrapper around proximity_analysis.py respecting its current CLI."""
    script_path = os.path.join(SCRIPT_DIR, 'proximity_analysis.py')

    cmd: List[str] = [sys.executable, script_path, str(erratic_id)]

    if update_db:
        cmd.append('--update-db')

    if output_file:
        cmd.extend(['--output', output_file])

    if verbose:
        cmd.append('--verbose')

    logger.info("Running proximity analysis for erratic %s", erratic_id)
    logger.debug("Command: %s", ' '.join(cmd))

    try:
        subprocess.run(cmd, check=True)
        logger.info("Proximity analysis for erratic %s completed successfully", erratic_id)
        return True
    except subprocess.CalledProcessError as exc:
        logger.error("Proximity analysis failed: %s", exc)
        return False

def run_classification(
    erratic_id: Optional[int] = None,
    build_topics: bool = False,
    update_db: bool = False,
    output_file: Optional[str] = None,
    verbose: bool = False,
    model_dir: Optional[str] = None,
):
    """Run the classify_erratic.py script with the appropriate flags.

    Args:
        erratic_id: ID of the erratic to classify.  Optional when *build_topics* is
            True.
        build_topics: When True the underlying script will build (or rebuild)
            the topic model from all erratic descriptions.
        update_db: Forwarded to the script's --update-db flag.
        output_file: Forwarded to the script's --output parameter.
        verbose: If True pass --verbose.
        model_dir: Optional custom directory for model files (passed through as
            --model-dir).
    """
    if erratic_id is None and not build_topics:
        raise ValueError("erratic_id is required unless build_topics is True")

    script_path = os.path.join(SCRIPT_DIR, 'classify_erratic.py')

    cmd: List[str] = [sys.executable, script_path]

    # Positional arg (erratic_id) comes *after* flags; argparse is order-agnostic.
    if build_topics:
        cmd.append('--build-topics')

    if erratic_id is not None:
        cmd.append(str(erratic_id))

    if update_db:
        cmd.append('--update-db')

    if output_file:
        cmd.extend(['--output', output_file])

    if model_dir:
        cmd.extend(['--model-dir', model_dir])

    if verbose:
        cmd.append('--verbose')

    human_readable_target = f"erratic {erratic_id}" if erratic_id is not None else "topic-model build"
    logger.info("Running classification: %s", human_readable_target)
    logger.debug("Command: %s", ' '.join(cmd))

    try:
        subprocess.run(cmd, check=True)
        logger.info("Classification task finished successfully: %s", human_readable_target)
        return True
    except subprocess.CalledProcessError as exc:
        logger.error("Classification task failed: %s", exc)
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
    

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Glacial Erratics Spatial Analysis Tools')
    subparsers = parser.add_subparsers(dest='command', help='Analysis command to run')
    
    # Proximity analysis parser
    proximity_parser = subparsers.add_parser('proximity', help='Run proximity analysis for an erratic')
    proximity_parser.add_argument('erratic_id', type=int, help='ID of the erratic to analyze')
    proximity_parser.add_argument('--update-db', action='store_true', help='Update database with results')
    proximity_parser.add_argument('--output', type=str, help='Output file for results (JSON)')
    proximity_parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    
    # Classification parser
    classify_parser = subparsers.add_parser('classify', help='Run classification for an erratic')
    classify_parser.add_argument('erratic_id', type=int, nargs='?', help='ID of the erratic to classify (omit when --build-topics)')
    classify_parser.add_argument('--build-topics', action='store_true', help='(Re)build topic model using all erratic descriptions')
    classify_parser.add_argument('--update-db', action='store_true', help='Update database with results')
    classify_parser.add_argument('--output', type=str, help='Output file for results (JSON)')
    classify_parser.add_argument('--model-dir', type=str, help='Custom directory for classifier model files')
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
    
    # Pipeline parser
    pipeline_parser = subparsers.add_parser('pipeline', help='Run full analysis pipeline')
    pipeline_parser.add_argument('--cluster-algorithm', type=str, default='dbscan', choices=['dbscan', 'kmeans', 'hierarchical'], help='Clustering algorithm for pipeline step (default: dbscan)')
    pipeline_parser.add_argument('--cluster-output', type=str, help='Output file for clustering results (default: data/results/pipeline_cluster_results.json)')
    pipeline_parser.add_argument('--classify-each', action='store_true', help='Run classification for every erratic')
    pipeline_parser.add_argument('--proximity-each', action='store_true', help='Run proximity analysis for every erratic')
    pipeline_parser.add_argument('--update-db', action='store_true', help='Update DB during per-erratic steps')
    pipeline_parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    # Pass-through clustering params
    pipeline_parser.add_argument('--k', type=int, help='K for KMeans/Hierarchical when pipeline clustering uses those algorithms')
    pipeline_parser.add_argument('--eps', type=float, help='Epsilon for DBSCAN when pipeline clustering uses DBSCAN')
    pipeline_parser.add_argument('--min-samples', type=int, dest='min_samples', help='min_samples for DBSCAN when used in pipeline')
    pipeline_parser.add_argument('--metric', type=str, choices=['auto', 'haversine', 'euclidean', 'cosine'], help='Distance metric for DBSCAN or clustering metric where applicable')
    pipeline_parser.add_argument('--linkage', type=str, choices=['ward', 'complete', 'average', 'single'], help='Linkage for Hierarchical clustering')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Run the requested command
    if args.command == 'proximity':
        success = run_proximity_analysis(
            args.erratic_id,
            args.update_db,
            args.output,
            args.verbose,
        )
    elif args.command == 'classify':
        try:
            success = run_classification(
                args.erratic_id,
                args.build_topics,
                args.update_db,
                args.output,
                args.verbose,
                args.model_dir,
            )
        except ValueError as err:
            logger.error(str(err))
            parser.error(str(err))
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
    elif args.command == 'pipeline' or args.command is None:
        # Gather clustering kwargs sans None values
        cluster_kwargs = {k: v for k, v in {
            'k': getattr(args, 'k', None),
            'eps': getattr(args, 'eps', None),
            'min_samples': getattr(args, 'min_samples', None),
            'metric': getattr(args, 'metric', None),
            'linkage': getattr(args, 'linkage', None),
        }.items() if v is not None}

        success = run_full_pipeline(
            cluster_algorithm=getattr(args, 'cluster_algorithm', 'dbscan'),
            cluster_output=getattr(args, 'cluster_output', None),
            classify_each=getattr(args, 'classify_each', False),
            proximity_each=getattr(args, 'proximity_each', False),
            update_db=getattr(args, 'update_db', False),
            verbose=getattr(args, 'verbose', False),
            **cluster_kwargs,
        )
    else:
        parser.print_help()
        return 0
    
    return 0 if success else 1

# ---------------------------------------------------------------------------
# FULL PIPELINE EXECUTION
# ---------------------------------------------------------------------------

def run_full_pipeline(
    *,
    cluster_algorithm: str = 'dbscan',
    cluster_output: Optional[str] = None,
    classify_each: bool = False,
    proximity_each: bool = False,
    update_db: bool = False,
    verbose: bool = False,
    **cluster_kwargs,
) -> bool:
    """Execute the full analysis pipeline.

    The pipeline consists of:
    1. Topic-model building (classification script with --build-topics).
    2. Clustering all erratics (algorithm configurable).
    3. Optional per-erratic classification and/or proximity analysis.

    Args:
        cluster_algorithm: Algorithm for clustering sub-step.
        cluster_output: Path to JSON file where clustering results will be stored.
                       If None, defaults to data/results/pipeline_cluster_results.json
        classify_each: If True run classification for every erratic in the DB.
        proximity_each: If True run proximity analysis for every erratic.
        update_db: Pass --update-db when running per-erratic steps.
        verbose: Propagate verbose flag to underlying scripts.
        **cluster_kwargs: Extra kwargs forwarded to run_clustering (e.g. eps, k, etc.).
    """
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Set default output path if not provided
    if cluster_output is None:
        cluster_output = get_default_output_path('pipeline_cluster_results.json')

    logger.info("=== PIPELINE STEP 1: Building topic model ===")
    if not run_classification(build_topics=True, verbose=verbose):
        logger.error("Pipeline aborted: topic-model build failed.")
        return False

    logger.info("=== PIPELINE STEP 2: Clustering erratics (%s) ===", cluster_algorithm)
    if not run_clustering(algorithm=cluster_algorithm, output_file=cluster_output, verbose=verbose, **cluster_kwargs):
        logger.error("Pipeline aborted: clustering failed.")
        return False

    # Import here to avoid circular import and heavy deps if not needed
    from utils import db_utils  # type: ignore

    erratic_ids: List[int] = []
    if classify_each or proximity_each:
        try:
            gdf = db_utils.load_all_erratics_gdf()
            erratic_ids = gdf['id'].dropna().astype(int).tolist()
            logger.info("Loaded %d erratic IDs for per-erratic analyses.", len(erratic_ids))
        except Exception as exc:
            logger.error("Failed to load erratic IDs from DB: %s", exc, exc_info=True)
            return False

    # Per-erratic classification
    if classify_each:
        logger.info("=== PIPELINE STEP 3: Classifying each erratic (%d total) ===", len(erratic_ids))
        for eid in erratic_ids:
            if not run_classification(erratic_id=eid, update_db=update_db, verbose=verbose):
                logger.warning("Classification failed for erratic %s; continuing.", eid)

    # Per-erratic proximity analysis
    if proximity_each:
        logger.info("=== PIPELINE STEP 4: Proximity analysis for each erratic (%d total) ===", len(erratic_ids))
        for eid in erratic_ids:
            if not run_proximity_analysis(erratic_id=eid, update_db=update_db, verbose=verbose):
                logger.warning("Proximity analysis failed for erratic %s; continuing.", eid)

    logger.info("=== PIPELINE COMPLETE ===")
    return True

if __name__ == "__main__":
    sys.exit(main()) 