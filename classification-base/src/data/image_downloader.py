import os
import time
import ssl
import glob
import yaml
from datetime import datetime
from pathlib import Path
from bing_image_downloader import downloader
import concurrent.futures
import sys

# Default configuration file to load
CONFIG_FILE = "cfg/macbook_config.yaml"
# CONFIG_FILE = "cfg/non_mac_config.yaml"


def load_config(config_file: str) -> dict:
    with open(config_file, 'r') as f:
        return yaml.safe_load(f)


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def download_images_for_query(query: str, root_dir: Path, n_images: int) -> dict:
    start_time = time.time()
    downloaded_count = 0
    clean_query = query.replace(" ", "_")

    print(f"[Thread: {clean_query}] Request started...")

    try:
        downloader.download(
            query=query,
            limit=n_images,
            output_dir=str(root_dir),
            adult_filter_off=True,
            force_replace=False,
            timeout=10,
            verbose=False
        )

        final_save_path = root_dir / query
        if final_save_path.exists():
            downloaded_count = len(list(final_save_path.glob('*')))

        end_time = time.time()

        print(f"[Thread: {clean_query}] Success. Downloaded: {downloaded_count} ({end_time - start_time:.2f} sec)")
        return {
            'query': query,
            'status': 'SUCCESS',
            'count': downloaded_count,
            'duration': end_time - start_time
        }

    except Exception as e:
        end_time = time.time()
        print(f"[Thread: {clean_query}] ERROR: {e} ({end_time - start_time:.2f} sec)")
        return {
            'query': query,
            'status': f'ERROR: {e.__class__.__name__}',
            'count': 0,
            'duration': end_time - start_time
        }


def main():
    config_file = CONFIG_FILE

    try:
        config = load_config(config_file)
    except FileNotFoundError:
        print(f"Error: Configuration file '{config_file}' not found.")
        sys.exit(1)
    except yaml.YAMLError as e:
        print(f"Error reading YAML configuration: {e}")
        sys.exit(1)

    TARGET_QUERIES = config['target_queries']
    N_IMAGES_PER_QUERY = config['n_images_per_query']
    MAX_WORKERS = config['max_workers']
    PROJECT_NAME = config['project_name']
    DATA_ROOT_DIR = config['data_root_dir']

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    unique_root_dir = Path(f"{DATA_ROOT_DIR}/{PROJECT_NAME}-{timestamp}")
    ensure_dir(unique_root_dir)

    try:
        ssl._create_default_https_context = ssl._create_unverified_context
        print("Temporary SSL fix applied to prevent CERTIFICATE_VERIFY_FAILED errors.")
    except AttributeError:
        pass

    print("\n" + "=" * 60)
    print("START MULTI-THREAD DOWNLOAD")
    print(f"Using configuration file: {config_file}")
    print(f"Queries in queue: {len(TARGET_QUERIES)}")
    print(f"Images per query: {N_IMAGES_PER_QUERY}")
    print(f"Max workers: {MAX_WORKERS}")
    print(f"Root save folder: {unique_root_dir.resolve()}")
    print("=" * 60 + "\n")

    overall_start_time = time.time()
    results = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_query = {
            executor.submit(
                download_images_for_query, query, unique_root_dir, N_IMAGES_PER_QUERY
            ): query
            for query in TARGET_QUERIES
        }

        for future in concurrent.futures.as_completed(future_to_query):
            query = future_to_query[future]
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                print(f"Critical error processing query {query}: {e}")
                results.append({'query': query, 'status': 'CRITICAL_ERROR', 'count': 0, 'duration': 0})

    overall_end_time = time.time()

    total_downloaded = sum(r['count'] for r in results)
    total_queries = len(TARGET_QUERIES)
    success_queries = sum(1 for r in results if r['status'] == 'SUCCESS')
    error_queries = total_queries - success_queries

    print("\n" + "=" * 60)
    print("GENERAL DOWNLOAD REPORT")
    print("=" * 60)
    print(f"Total elapsed time: {overall_end_time - overall_start_time:.2f} seconds")
    print(f"Total queries processed: {total_queries}")
    print(f"Successfully completed: {success_queries}")
    print(f"Completed with errors: {error_queries}")
    print(f"TOTAL IMAGES DOWNLOADED: {total_downloaded}")
    print("=" * 60)

    print("\nDETAILED QUERY REPORT:")
    for result in results:
        print(f"  [{result['status']}] {result['query']}: {result['count']} files ({result['duration']:.2f} sec)")


if __name__ == "__main__":
    main()