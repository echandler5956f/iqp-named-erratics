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

def _execute_analysis_script(
    script_name: str,
    script_args: List[str],
    log_action_description: str,
) -> bool:
    """
    Helper to construct and run an analysis script as a subprocess.

    Args:
        script_name: The filename of the Python script to run (e.g., 'proximity_analysis.py').
        script_args: A list of string arguments to pass to the script.
        log_action_description: A human-readable description of the action for logging.

    Returns:
        True if the script ran successfully, False otherwise.
    """
    full_script_path = os.path.join(SCRIPT_DIR, script_name)
    # Ensure all script_args are strings
    cmd: List[str] = [sys.executable, full_script_path] + [str(arg) for arg in script_args]

    logger.info("Executing: %s", log_action_description)
    logger.debug("Full command: %s", ' '.join(cmd))

    try:
        # Scripts are expected to print JSON to stdout for the Node.js service.
        # Thus, capture_output should be False or handled carefully if True.
        # check=True will raise CalledProcessError on non-zero exit codes.
        # text=True decodes stdout/stderr as text.
        process = subprocess.run(cmd, check=True, text=True, capture_output=False)
        logger.info("%s completed successfully.", log_action_description)
        return True
    except subprocess.CalledProcessError as exc:
        logger.error(
            "%s failed. Subprocess returned non-zero exit status %d.",
            log_action_description, exc.returncode
        )
        # Stderr and stdout are not captured if capture_output=False,
        # but the called script's output will go to the parent's stderr/stdout.
        # If capture_output were True:
        # if exc.stderr:
        #     logger.error("Script stderr:\\n%s", exc.stderr.strip())
        # if exc.stdout:
        #     logger.error("Script stdout (on error):\\n%s", exc.stdout.strip())
        return False
    except FileNotFoundError:
        logger.error("Failed to run %s: Script not found at %s.", log_action_description, full_script_path)
        return False
    except Exception as e: # Catch any other potential errors during subprocess setup/execution
        logger.error("An unexpected error occurred while trying to run %s: %s", log_action_description, e, exc_info=True)
        return False

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
    script_args: List[str] = [str(erratic_id)]

    if update_db:
        script_args.append('--update-db')
    if output_file:
        script_args.extend(['--output', output_file])
    if verbose:
        script_args.append('--verbose')
    
    return _execute_analysis_script(
        script_name='proximity_analysis.py',
        script_args=script_args,
        log_action_description=f"Proximity analysis for erratic ID {erratic_id}"
    )

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

    script_args: List[str] = [] # Using Any temporarily as elements can be int or str before conversion in helper

    if build_topics:
        script_args.append('--build-topics')
    if erratic_id is not None: # erratic_id is positional, should come after flags if argparse handles it that way
        script_args.append(erratic_id) # Script expects ID as positional arg
    if update_db:
        script_args.append('--update-db')
    if output_file:
        script_args.extend(['--output', output_file])
    if model_dir:
        script_args.extend(['--model-dir', model_dir])
    if verbose:
        script_args.append('--verbose')

    human_readable_target = f"erratic ID {erratic_id}" if erratic_id is not None else "topic model build"
    log_description = f"Classification task for {human_readable_target}"
    
    # classify_erratic.py might have specific argument order needs.
    # The original code had: cmd = [sys.executable, script_path]
    # then: if build_topics: cmd.append('--build-topics')
    # then: if erratic_id is not None: cmd.append(str(erratic_id))
    # This suggests flags first, then positional.
    # The _execute_analysis_script helper adds sys.executable and script_path.
    # We need to ensure script_args are passed in the correct order if the target script is sensitive.
    # For classify_erratic.py: flags, then optional erratic_id.

    final_script_args = []
    if build_topics:
        final_script_args.append('--build-topics')
    # All other flags before positional erratic_id
    if update_db:
        final_script_args.append('--update-db')
    if output_file:
        final_script_args.extend(['--output', output_file])
    if model_dir:
        final_script_args.extend(['--model-dir', model_dir])
    if verbose:
        final_script_args.append('--verbose')
    if erratic_id is not None:
        final_script_args.append(str(erratic_id))

    return _execute_analysis_script(
        script_name='classify_erratic.py',
        script_args=final_script_args,
        log_action_description=log_description
    )

def run_clustering(algorithm: str, output_file: str, verbose: bool = False, 
                   k: Optional[int] = None, features: Optional[List[str]] = None, 
                   eps: Optional[float] = None, min_samples: Optional[int] = None, 
                   metric: Optional[str] = None, linkage: Optional[str] = None):
    """Run clustering analysis for all erratics using the specified algorithm and parameters."""
    script_args: List[str] = ['--algorithm', algorithm, '--output', output_file]
    
    if k is not None:
        script_args.extend(['--k', str(k)])
    if features:
        script_args.extend(['--features'] + features)
    if eps is not None:
        script_args.extend(['--eps', str(eps)])
    if min_samples is not None:
        script_args.extend(['--min_samples', str(min_samples)])
    if metric:
        script_args.extend(['--metric', metric])
    if linkage:
        script_args.extend(['--linkage', linkage])
    if verbose:
        script_args.append('--verbose')
    
    return _execute_analysis_script(
        script_name='clustering.py',
        script_args=script_args,
        log_action_description=f"Clustering analysis ({algorithm}) for all erratics"
    )
    

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Glacial Erratics Spatial Analysis Tools')
    subparsers = parser.add_subparsers(dest='command', help='Analysis command to run')
    
    # Proximity analysis parser
    proximity_parser = subparsers.add_parser('proximity', help='Run proximity analysis for a single specified erratic.')
    proximity_parser.add_argument('erratic_id', type=int, help='ID of the erratic to analyze.')
    proximity_parser.add_argument('--update-db', action='store_true', help='Update the database with analysis results.')
    proximity_parser.add_argument('--output', type=str, help='Optional: Path to JSON file to save analysis results.')
    proximity_parser.add_argument('--verbose', '-v', action='store_true', help='Enable detailed console output.')
    
    # Classification parser
    classify_parser = subparsers.add_parser('classify', help='Run NLP classification for an erratic or build topic models.')
    classify_parser.add_argument('erratic_id', type=int, nargs='?', help='ID of the erratic to classify. Omit this when using --build-topics.')
    classify_parser.add_argument('--build-topics', action='store_true', help='(Re)build topic models using all erratic descriptions from the database.')
    classify_parser.add_argument('--update-db', action='store_true', help='Update the database with classification results (applies if erratic_id is provided).')
    classify_parser.add_argument('--output', type=str, help='Optional: Path to JSON file to save classification results or build log.')
    classify_parser.add_argument('--model-dir', type=str, help='Optional: Custom directory for storing/loading classifier model files.')
    classify_parser.add_argument('--verbose', '-v', action='store_true', help='Enable detailed console output.')
    
    # Clustering parser
    cluster_parser = subparsers.add_parser('cluster', help='Run spatial clustering analysis for all erratics.')
    cluster_parser.add_argument('--algorithm', type=str, default='dbscan', choices=['dbscan', 'kmeans', 'hierarchical'], help='Clustering algorithm to use (default: dbscan).')
    cluster_parser.add_argument('--output', type=str, required=True, help='Path to JSON file to save clustering results.')
    # K-Means / Hierarchical args
    cluster_parser.add_argument('--k', type=int, help='K-Means/Hierarchical: Number of clusters (K). Required for these algorithms.')
    cluster_parser.add_argument('--features', nargs='+', help='Features for clustering (e.g., longitude latitude). Default varies by algorithm. Provide as space-separated list.')
    cluster_parser.add_argument('--linkage', type=str, choices=['ward', 'complete', 'average', 'single'], default='ward', help='Hierarchical: Linkage criterion (default: ward).')
    # DBSCAN args
    cluster_parser.add_argument('--eps', type=float, help='DBSCAN: Epsilon parameter (search radius). Auto-estimated if not provided.')
    cluster_parser.add_argument('--min_samples', type=int, default=3, help='DBSCAN: Minimum number of samples in a neighborhood for a point to be considered as a core point (default: 3).')
    cluster_parser.add_argument('--metric', type=str, default='haversine', choices=['auto', 'haversine', 'euclidean', 'cosine'], help='DBSCAN/Clustering: Distance metric to use (default: haversine for geo-data). \'auto\' lets the algorithm decide.')
    cluster_parser.add_argument('--verbose', '-v', action='store_true', help='Enable detailed console output.')
    
    # Pipeline parser
    pipeline_parser = subparsers.add_parser('pipeline', help='Run the full multi-step analysis pipeline (topic modeling, clustering, and optional per-erratic analyses).')
    pipeline_parser.add_argument('--cluster-algorithm', type=str, default='dbscan', choices=['dbscan', 'kmeans', 'hierarchical'], help='Pipeline Step: Clustering algorithm (default: dbscan).')
    pipeline_parser.add_argument('--cluster-output', type=str, help='Pipeline Step: Output JSON file for clustering results (default: data/results/pipeline_cluster_results.json).')
    pipeline_parser.add_argument('--classify-each', action='store_true', help='Pipeline Step: Run classification for every erratic in the database.')
    pipeline_parser.add_argument('--proximity-each', action='store_true', help='Pipeline Step: Run proximity analysis for every erratic in the database.')
    pipeline_parser.add_argument('--update-db', action='store_true', help='Pipeline Step: Update database during per-erratic classification/proximity steps.')
    pipeline_parser.add_argument('--verbose', '-v', action='store_true', help='Enable detailed console output for all pipeline steps.')
    # Pass-through clustering params for pipeline
    pipeline_parser.add_argument('--k', type=int, help='Pipeline Clustering: K for KMeans/Hierarchical.')
    pipeline_parser.add_argument('--eps', type=float, help='Pipeline Clustering: Epsilon for DBSCAN.')
    pipeline_parser.add_argument('--min-samples', type=int, dest='min_samples_pipeline', help='Pipeline Clustering: min_samples for DBSCAN (use min_samples_pipeline to avoid conflict with top-level cluster command).')
    pipeline_parser.add_argument('--metric', type=str, choices=['auto', 'haversine', 'euclidean', 'cosine'], dest='metric_pipeline', help='Pipeline Clustering: Distance metric for DBSCAN or applicable algorithms (use metric_pipeline to avoid conflict).')
    pipeline_parser.add_argument('--linkage', type=str, choices=['ward', 'complete', 'average', 'single'], dest='linkage_pipeline', help='Pipeline Clustering: Linkage for Hierarchical clustering (use linkage_pipeline to avoid conflict).')
    pipeline_parser.add_argument('--max-workers', type=int, help='Pipeline Parallelism: Maximum parallel workers for per-erratic steps (default: 2xCPU count, capped at 32).')
    
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
            'k': getattr(args, 'k', None), # Will take pipeline --k if specified, else None
            'eps': getattr(args, 'eps', None), # Will take pipeline --eps if specified, else None
            'min_samples': getattr(args, 'min_samples_pipeline', None), # Use the renamed pipeline arg
            'metric': getattr(args, 'metric_pipeline', None),          # Corrected: Use the renamed pipeline arg
            'linkage': getattr(args, 'linkage_pipeline', None),        # Corrected: Use the renamed pipeline arg
        }.items() if v is not None}

        max_workers_override = getattr(args, 'max_workers', None)

        success = run_full_pipeline(
            cluster_algorithm=getattr(args, 'cluster_algorithm', 'dbscan'),
            cluster_output=getattr(args, 'cluster_output', None),
            classify_each=getattr(args, 'classify_each', False),
            proximity_each=getattr(args, 'proximity_each', False),
            update_db=getattr(args, 'update_db', False),
            verbose=getattr(args, 'verbose', False),
            max_workers=max_workers_override,
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
    max_workers: Optional[int] = None,
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
        max_workers: Maximum parallel workers for per-erratic steps.
        **cluster_kwargs: Extra kwargs forwarded to run_clustering (e.g. eps, k, etc.).
    """
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Set default output path if not provided
    if cluster_output is None:
        cluster_output = get_default_output_path('pipeline_cluster_results.json')

    # Prefetch all data sources before parallel processing
    if proximity_each and max_workers and max_workers > 1:
        logger.info("=== PREFETCHING DATA SOURCES ===")
        try:
            from data_pipeline import prefetch_all
            prefetch_all()
            logger.info("Data sources prefetched successfully")
        except Exception as e:
            logger.warning(f"Failed to prefetch data sources: {e}. Continuing anyway...")

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

    # Determine worker pool size once.
    if max_workers is None or max_workers <= 0:
        import multiprocessing
        max_workers = min(32, (multiprocessing.cpu_count() or 1) * 2)

    from concurrent.futures import ThreadPoolExecutor, as_completed

    def _parallel_run(func, ids, label):
        if not ids:
            return
        # For memory-heavy proximity jobs, clamp workers. This value (10) is an empirical balance.
        # Original thought was 4, but 10 seems to be the current value from a previous observation.
        # This helps prevent OOM errors when many proximity analyses run concurrently.
        workers = max_workers if label != "Proximity analysis" else min(max_workers, 10)
        logger.info("=== PIPELINE STEP: %s (%d tasks, %d workers) ===", label, len(ids), workers)
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(func, eid): eid for eid in ids}
            for fut in as_completed(futures):
                eid = futures[fut]
                try:
                    ok = fut.result()
                    if not ok:
                        logger.warning("%s failed for erratic %s", label, eid)
                except Exception as exc:
                    logger.error("%s raised for erratic %s: %s", label, eid, exc)

    if classify_each:
        _parallel_run(lambda eid: run_classification(erratic_id=eid, update_db=update_db, verbose=verbose), erratic_ids, "Classification")

    if proximity_each:
        # Use in-process calculation when single worker to avoid heavy subprocess overhead
        # and to make debugging/profiling of a single proximity run easier if needed.
        if max_workers == 1:
            from proximity_analysis import calculate_proximity  # noqa: WPS433

            def _prox(eid):
                res = calculate_proximity(eid)
                if update_db and 'proximity_analysis' in res:
                    from utils import db_utils  # local import to avoid early heavy deps
                    db_utils.update_erratic_analysis_results(eid, res['proximity_analysis'])
                return 'error' not in res

            _parallel_run(_prox, erratic_ids, "Proximity analysis")
        else:
            _parallel_run(lambda eid: run_proximity_analysis(erratic_id=eid, update_db=update_db, verbose=verbose), erratic_ids, "Proximity analysis")

    logger.info("=== PIPELINE COMPLETE ===")
    return True

if __name__ == "__main__":
    sys.exit(main()) 