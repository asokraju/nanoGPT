import os
import pandas as pd
import logging
from typing import List, Dict, Any, Union
from omegaconf import ListConfig

logger = logging.getLogger(__name__)













def process_csv_file(preprocess_cfg: Dict, row_selection: Union[str, List[str]] = "all") -> pd.DataFrame:
    """
    Reads a CSV file and constructs per-dataset paths. Filters rows based on data_dir_unique instead of row numbers.
    """
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    input_csv_path = preprocess_cfg.get("input_csv_path", "")
    root_dir = preprocess_cfg.get("root_dir", "").strip()

    logger.info(f"Loading CSV file: {input_csv_path}")
    try:
        df = pd.read_csv(input_csv_path)
        logger.info("CSV file loaded successfully.")
    except FileNotFoundError as exc:
        logger.error(f"CSV file not found: {input_csv_path} - {exc}")
        raise
    except Exception as exc:
        logger.exception(f"Error reading CSV file: {exc}")
        raise
    
    # Ensure the required column exists
    if "data_dir_unique" not in df.columns:
        raise ValueError("CSV must contain a 'data_dir_unique' column.")
    
    # Filter rows by data_dir_unique if row_selection != "all"
    if isinstance(row_selection, str):
        if row_selection.lower() == "all":
            logger.info("Processing ALL rows (row_selection='all').")
        else:

            logger.info(f"Processing only one dataset: {row_selection}")
            df = df[df["data_dir_unique"] == row_selection]
    elif isinstance(row_selection, list) and len(row_selection) > 0:
        logger.info(f"Processing these data_dir_unique values: {row_selection}")
        df = df[df["data_dir_unique"].isin(row_selection)]
    else:
        logger.warning("row_selection is invalid or empty. Processing ALL rows.")
    
    # Create a dataset_id column (copy from data_dir_unique)
    df["dataset_id"] = df["data_dir_unique"]
    
    # Build subdirectory paths based on root_dir and data_dir_unique
    df["filepath_ground_truth"] = df["data_dir_unique"].apply(
        lambda x: os.path.join(root_dir, x, "ground_truth")
    )
    df["filepath_source_docs"] = df["data_dir_unique"].apply(
        lambda x: os.path.join(root_dir, x, "source_docs")
    )
    df["filepath_intermedia_context_files"] = df["data_dir_unique"].apply(
        lambda x: os.path.join(root_dir, x, "intermedia_context_files")
    )
    df["filepath_llm_responses"] = df["data_dir_unique"].apply(
        lambda x: os.path.join(root_dir, x, "llm_responses")
    )
    df["filepath_metrics"] = df["data_dir_unique"].apply(
        lambda x: os.path.join(root_dir, x, "metrics")
    )
    
    logger.info("Finished processing CSV file and constructing directory paths.")
    return df





def get_meta_data(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Converts a DataFrame into a list of dictionaries, each containing metadata for datasets.
    """

    
    
    
    
    
    
    
    
    
    
    
    result = []
    logger.info("Extracting metadata from DataFrame.")

    # Check for required columns
    required_columns = ["filepath_source_docs", "filepath_ground_truth"]
    for col in required_columns:
        if col not in df.columns:
            logger.warning(f"Column {col} not found in DataFrame. Metadata may be incomplete.")

    for _, row in df.iterrows():
        dataset_id = row["dataset_id"]
        source_docs_dir = str(row.get("filepath_source_docs", "")).strip()
        ground_truth_dir = str(row.get("filepath_ground_truth", "")).strip()

        # List files in source_docs_dir
        documents = []
        if source_docs_dir and os.path.isdir(source_docs_dir):
            try:
                documents = [
                    os.path.join(source_docs_dir, f)
                    for f in os.listdir(source_docs_dir)
                    if os.path.isfile(os.path.join(source_docs_dir, f)) and not f.startswith('.')
                ]
            except FileNotFoundError:
                logger.warning(f"[{dataset_id}] Source docs directory not found: {source_docs_dir}")
        else:
            logger.warning(f"[{dataset_id}] Source docs directory missing or invalid: {source_docs_dir}")

        # Attempt to find a ground truth file
        ground_truth_file = None
        if ground_truth_dir and os.path.isdir(ground_truth_dir):
            potential_gt_files = [f for f in os.listdir(ground_truth_dir) if not f.startswith('.')]
            if potential_gt_files:
                potential_gt = os.path.join(ground_truth_dir, potential_gt_files[0])
                if os.path.isfile(potential_gt):
                    ground_truth_file = potential_gt
            else:
                logger.warning(f"[{dataset_id}] No ground truth found in {ground_truth_dir}")
        else:
            logger.warning(f"[{dataset_id}] No files available in {ground_truth_dir} to process")

        # Build metadata dictionary
        item = {
            **row.to_dict(),  # All row data
            "documents": documents,
            "ground_truth_file": ground_truth_file,
        }
        result.append(item)

    logger.info("Metadata extraction complete.")
    return result


import os
import time
import logging
import pandas as pd
from typing import Any, Dict, List

# Setup logger
logger = logging.getLogger(__name__)

# =============================================================================
# Upload Stage
# =============================================================================
def dataset_upload(meta_data_list: List[Dict[str, Any]], cli: SparkCli, upload_cfg: dict) -> None:
    """
    Processes documents referenced in each metadata item in meta_data_list using a direct query
    on the full_document_path. For each file:
      - Normalize the file path.
      - Query the system for a document with the matching full_document_path.
      - If more than one document is found, log a warning and skip upload for that file.
      - If exactly one document is found:
          * If reupload is True, delete the document and mark for re-upload.
          * If reupload is False, skip uploading this file.
      - If no document is found, mark the file for upload.
    After processing all files for an item, upload any files marked for upload.
    
    Upload configuration (upload_cfg) is expected to contain:
        reupload (bool): If True, delete and re-upload an existing document.
    
    Args:
        meta_data_list (List[Dict]): Output of get_meta_data (each item contains a "documents" list and "tag" dictionary).
        cli (SparkCli): The CLI client for uploading/deleting documents.
        upload_cfg (dict): Upload configuration dictionary.
    """
    logger.info("Starting upload stage with metadata using direct full_document_path queries.")
    reupload = upload_cfg.get("reupload", False)

    for item in meta_data_list:
        documents = item.get("documents", [])
        base_tags = item.get("tag", {})
        dataset_id = base_tags.get("dataset_id", "N/A")
        # List to hold the normalized file paths that need to be uploaded.
        to_upload = []

        for doc in documents:
            norm_doc = os.path.normpath(doc.strip())
            # Ensure file exists locally
            if not os.path.isfile(norm_doc):
                logger.warning(f"File does not exist: {norm_doc}")
                continue

            # Query the system for this specific full_document_path.
            try:
                query = {"any.tags.full_document_path": norm_doc}
                results = cli.documents(query_terms=query)
            except Exception as e:
                logger.exception(f"Error querying for {norm_doc}: {e}")
                continue

            num_results = len(results)
            logger.debug(f"Query for {norm_doc} returned {num_results} results.")

            if num_results > 1:
                logger.warning(f"Duplicate documents found for {norm_doc} (found {num_results}). Skipping upload.")
                continue  # Skip this file entirely to avoid duplicates.
            elif num_results == 1:
                # Exactly one document exists.
                if reupload:
                    existing_doc = results[0]
                    existing_doc_id = existing_doc.get("id")
                    logger.info(f"Re-upload enabled: Deleting existing doc_id={existing_doc_id} for file {norm_doc}.")
                    try:
                        cli.delete_document(existing_doc_id)
                    except Exception as ex:
                        logger.error(f"Error deleting doc {existing_doc_id} for file {norm_doc}: {ex}")
                        continue  # Skip upload for this file if deletion fails.
                    to_upload.append(norm_doc)
                else:
                    logger.info(f"Skipping upload for {norm_doc}; already present in the system.")
            else:
                # No document found; mark file for upload.
                logger.info(f"File {norm_doc} not found in system; marking for upload.")
                to_upload.append(norm_doc)

        if not to_upload:
            logger.info(f"No new documents to upload for dataset_id {dataset_id}.")
            continue

        # Prepare tags for each file to be uploaded.
        tags_for_upload = []
        for doc_path in to_upload:
            # Make a shallow copy of the base tag and add/update the full_document_path.
            doc_tags = dict(base_tags)
            doc_tags["full_document_path"] = doc_path
            tags_for_upload.append(doc_tags)

        # Upload the documents.
        logger.info(f"Uploading {len(to_upload)} documents for dataset_id {dataset_id}.")
        try:
            uploaded_ids = cli.upload_documents(file_list=to_upload, tags=tags_for_upload)
            logger.info(f"Uploaded documents for dataset_id {dataset_id}: {uploaded_ids}")
        except Exception as exc:
            logger.exception(f"Error uploading documents for dataset_id {dataset_id}: {exc}")

# =============================================================================
# Status Stage
# =============================================================================
def dataset_ensure_ready(cli: SparkCli, meta_data_list: List[Dict[str, Any]], status_cfg: dict) -> None:
    """
    Checks the status of documents corresponding to the processed rows (meta_data_list).

    For each metadata item (representing a processed row), this function:
      - Iterates over each file path in the "documents" list.
      - Normalizes the file path.
      - Queries the document system using a direct query on "any.tags.full_document_path".
      - If exactly one document is found, logs its status.
      - If multiple documents are found, logs a warning (duplicates).
      - If no document is found, logs a warning.
      - Retries the query up to 'retry' times (with a delay of 'retry_time' seconds) upon transient errors.

    At the end, it logs a summary of:
      - Total files checked
      - Number of files not found
      - Number of duplicate results
      - Number of documents with status "ready"
      - Number of documents that are found but not ready
      - Number of documents for which errors occurred

    Status configuration (status_cfg) is expected to contain:
        retry (int): Number of times to retry status checks for each document.
        retry_time (int): Time in seconds to wait between retries.

    Args:
        cli (SparkCli): CLI client to get document status.
        meta_data_list (List[Dict[str, Any]]): List of metadata items corresponding to the rows to process.
        status_cfg (dict): Status configuration dictionary.
    """
    logger.info("Starting status stage for processed rows.")
    retry = status_cfg.get("retry", 3)
    retry_time = status_cfg.get("retry_time", 30)

    # Initialize summary counters
    total_files = 0
    not_found_count = 0
    duplicate_count = 0
    ready_count = 0
    not_ready_count = 0
    error_count = 0

    for item in meta_data_list:
        dataset_id = item.get("tag", {}).get("dataset_id", "N/A")
        documents = item.get("documents", [])
        logger.info(f"Checking documents for dataset_id {dataset_id}, number of files: {len(documents)}")
        for doc in documents:
            total_files += 1
            norm_doc = os.path.normpath(doc.strip())
            attempt = 0
            status_found = None
            # Retry loop for each document query
            while attempt < retry:
                attempt += 1
                try:
                    query = {"any.tags.full_document_path": norm_doc}
                    results = cli.documents(query_terms=query)
                    num_results = len(results)
                    if num_results == 0:
                        logger.warning(f"[{dataset_id}] No document found for file: {norm_doc}")
                        status_found = None
                        break
                    elif num_results > 1:
                        logger.warning(f"[{dataset_id}] Duplicate documents found for file: {norm_doc} (found {num_results}).")
                        status_found = "duplicate"
                        break
                    else:
                        # Exactly one document was found.
                        doc_status = results[0].get("status", "unknown")
                        logger.info(f"[{dataset_id}] File: {norm_doc} -> doc_id: {results[0].get('id')} status: {doc_status}")
                        status_found = doc_status
                        break
                except Exception as exc:
                    logger.exception(f"[{dataset_id}] Error fetching status for {norm_doc} (attempt {attempt}): {exc}")
                    if attempt < retry:
                        logger.info(f"[{dataset_id}] Retrying in {retry_time} seconds...")
                        time.sleep(retry_time)
                    else:
                        logger.error(f"[{dataset_id}] Giving up on {norm_doc} after {attempt} attempts.")
                        error_count += 1
                        status_found = "error"
                        break

            # Update summary counters based on the result.
            if status_found is None:
                not_found_count += 1
            elif status_found == "duplicate":
                duplicate_count += 1
            elif status_found == "ready":
                ready_count += 1
            elif status_found == "error":
                error_count += 1
            else:
                # Any status that is not "ready" is counted as not ready.
                not_ready_count += 1

    # Log summary statistics.
    logger.info("Status Stage Summary:")
    logger.info(f"Total files checked: {total_files}")
    logger.info(f"Documents not found: {not_found_count}")
    logger.info(f"Duplicate documents: {duplicate_count}")
    logger.info(f"Documents with status 'ready': {ready_count}")
    logger.info(f"Documents found but not ready: {not_ready_count}")
    logger.info(f"Documents with errors: {error_count}")

    # Also print the summary
    print(f"Total files checked: {total_files}")
    print(f"Documents not found: {not_found_count}")
    print(f"Duplicate documents: {duplicate_count}")
    print(f"Documents with status 'ready': {ready_count}")
    print(f"Documents found but not ready: {not_ready_count}")
    print(f"Documents with errors: {error_count}")

# =============================================================================
# Delete Stage
# =============================================================================
def dataset_delete(cli: SparkCli, df: pd.DataFrame) -> None:
    """
    Deletes documents based on the 'dataset_id' field present in the processed Excel DataFrame.

    For each unique dataset_id in the DataFrame, this function:
      - Queries the system for documents that have a tag 'dataset_id' matching the value.
      - Deletes each matching document.
    
    Args:
        cli (SparkCli): The CLI client to delete documents.
        df (pd.DataFrame): The processed Excel DataFrame containing a 'dataset_id' column.
    """
    logger.info("Starting delete stage using processed Excel data (row_to_process).")
    if "dataset_id" not in df.columns:
        logger.error("Column 'dataset_id' not found in DataFrame. Aborting delete stage.")
        return

    # Get unique dataset IDs from the DataFrame
    unique_dataset_ids = df["dataset_id"].unique()
    logger.info(f"Found {len(unique_dataset_ids)} unique dataset IDs: {unique_dataset_ids}")

    # Iterate over each unique dataset_id and delete matching documents.
    for dataset_id in unique_dataset_ids:
        logger.info(f"Deleting documents for dataset_id: {dataset_id}")
        try:
            matching_docs = cli.documents(query_terms={"any.tags.dataset_id": dataset_id})
            logger.info(f"Found {len(matching_docs)} documents for dataset_id: {dataset_id}")
            for doc in matching_docs:
                doc_id = doc.get("id")
                resp = cli.delete_document(doc_id)
                logger.info(f"Deleted doc_id={doc_id} for dataset_id {dataset_id} => {resp}")
        except Exception as exc:
            logger.exception(f"Error deleting documents for dataset_id {dataset_id}: {exc}")

# =============================================================================
# Retrieval Stage
# =============================================================================
def dataset_retrieval(cli: SparkCli, df: pd.DataFrame, retrieval_cfg: dict) -> None:
    """
    For each unique dataset in the processed Excel DataFrame (df), this function:
      - Determines the ground-truth file from the 'filepath_ground_truth' and 'filename_ground_truth' columns.
      - Loads the ground-truth Excel file.
      - Retrieves document IDs for that dataset using the dataset_id tag.
      - (Optionally) Checks that all required documents are in "ready" status.
      - For each row in the ground-truth file (processed sequentially), retrieves search results using:
            * If retrieval_cfg["limit"] is 0, the function will keep trying until a non-empty result is obtained.
            * If retrieval_cfg["limit"] is a non-zero integer (say X), it will try up to X attempts.
      - Saves the updated ground-truth DataFrame (with a new "search_output" column) to a CSV file.
        The output CSV filename is constructed from the ground-truth file's name with a "_retrieved" suffix
        and saved to retrieval_cfg["evaluation_output_dir"].
    
    Retrieval configuration (retrieval_cfg) is expected to contain:
        check_status (bool): Verify that documents are 'ready'
        query_col (str): Column name for the query text (e.g., "Question")
        limit (int): Number of attempts (if non-zero) or infinite if 0
        evaluation_output_dir (str): Output directory for the CSV file
        overwrite_existing (bool): Whether to overwrite existing CSV files
    
    Args:
        cli (SparkCli): Client interface for document and search API operations.
        df (pd.DataFrame): Processed Excel DataFrame containing at least:
                          'dataset_id', 'filepath_ground_truth', and 'filename_ground_truth'
        retrieval_cfg (dict): Retrieval configuration dictionary.
    """
    logger.info("Starting retrieval stage (sequential LLM calls).")
    check_status = retrieval_cfg.get("check_status", True)
    query_col = retrieval_cfg.get("query_col", "Question")
    limit = retrieval_cfg.get("limit", 20)
    eval_output_dir = retrieval_cfg.get("evaluation_output_dir", "./output")
    overwrite_existing = retrieval_cfg.get("overwrite_existing", True)
    os.makedirs(eval_output_dir, exist_ok=True)
    
    summary = {}
    unique_dataset_ids = df["dataset_id"].unique()
    logger.info(f"Found {len(unique_dataset_ids)} unique dataset IDs: {unique_dataset_ids}")

    for dataset_id in unique_dataset_ids:
        logger.info(f"Processing dataset_id: {dataset_id}")
        try:
            row = df[df["dataset_id"] == dataset_id].iloc[0]
        except Exception as e:
            logger.error(f"Error retrieving row for dataset_id {dataset_id}: {e}")
            summary[dataset_id] = f"Error retrieving row: {e}"
            continue

        # Build ground truth file path
        ground_truth_file = None
        gt_path = str(row.get("filepath_ground_truth", "")).strip()
        try:
            potential_gt_files = [f for f in os.listdir(gt_path) if not f.startswith('.')]
            if potential_gt_files:
                potential_gt = os.path.join(gt_path, potential_gt_files[0])
                if os.path.isfile(potential_gt):
                    ground_truth_file = potential_gt
                else:
                    logger.warning(f"[{dataset_id}] No valid ground truth file found in {gt_path}")
            else:
                logger.warning(f"[{dataset_id}] No files available in {gt_path} to process")
        except Exception as e:
            logger.error(f"Ground truth path not found: {gt_path}: {e}")
            summary[dataset_id] = f"Error accessing files: {e}"
            continue

        if not ground_truth_file or not os.path.isfile(ground_truth_file):
            logger.warning(f"Ground truth file not found for dataset_id {dataset_id}: {ground_truth_file}")
            summary[dataset_id] = "Ground truth file not found"
            continue

        # Load the ground truth Excel file
        try:
            ground_truth_df = pd.read_excel(ground_truth_file)
            logger.info(f"Loaded ground truth for dataset_id {dataset_id} from {ground_truth_file}")
        except Exception as e:
            logger.error(f"Error loading ground truth file for dataset_id {dataset_id}: {e}")
            summary[dataset_id] = f"Error loading ground truth: {e}"
            continue

        # Retrieve document IDs using the dataset_id tag
        try:
            docs_in_system = cli.documents(query_terms={"any.tags.dataset_id": dataset_id})
            document_ids = [doc["id"] for doc in docs_in_system if "id" in doc]
        except Exception as e:
            logger.error(f"[{dataset_id}] Error retrieving documents: {e}")
            summary[dataset_id] = f"Error retrieving docs: {e}"
            continue

        if not document_ids:
            logger.warning(f"[{dataset_id}] No documents found in Spark.")
            summary[dataset_id] = "No docs found in Spark"
            continue

        # Check that all docs are "ready" if required
        if check_status:
            not_ready_docs = [doc["id"] for doc in docs_in_system if doc.get("status") != "ready"]
            if not_ready_docs:
                logger.warning(f"Documents not ready for dataset_id {dataset_id}: {not_ready_docs}. Skipping.")
                summary[dataset_id] = f"Documents not ready: {not_ready_docs}"
                continue

        # Construct output CSV filename
        base_filename = os.path.splitext(potential_gt_files[0])[0]
        output_csv_filename = f"{base_filename}_retrieved.csv"
        output_csv_path = os.path.join(eval_output_dir, output_csv_filename)
        if not overwrite_existing and os.path.exists(output_csv_path):
            logger.info(f"Output CSV {output_csv_path} already exists; skipping dataset_id {dataset_id}.")
            summary[dataset_id] = "Output file exists; skipped"
            continue

        if query_col not in ground_truth_df.columns:
            logger.error(f"Query column '{query_col}' not found in ground truth for dataset_id {dataset_id}. Skipping.")
            summary[dataset_id] = f"Query column '{query_col}' missing"
            continue

        complete_success = True
        search_outputs = []

        for idx, gt_row in ground_truth_df.iterrows():
            query_text = str(gt_row.get(query_col, "")).strip()
            if not query_text:
                logger.warning(f"Empty query in row {idx} for dataset_id {dataset_id}.")
                search_outputs.append([])
                complete_success = False
                continue

            results = []
            if limit == 0:
                # limit 0 implies do it indefinitely
                attempt = 0
                while True:
                    attempt += 1
                    try:
                        results = cli.search_v2(query=query_text, document_ids=document_ids, limit=20)
                    except Exception as e:
                        logger.error(f"Error in search_v2 for query '{query_text}' (attempt {attempt}): {e}")
                        results = []
                    if results:
                        break
                    else:
                        logger.info(f"No results for query '{query_text}' (attempt {attempt}). Retrying after delay...")
                        time.sleep(10)
            else:
                for attempt in range(1, limit + 1):
                    try:
                        results = cli.search_v2(query=query_text, document_ids=document_ids, limit=20)
                    except Exception as e:
                        logger.error(f"Error in search_v2 for query '{query_text}' (attempt {attempt}): {e}")
                        results = []
                    if results:
                        break
                    else:
                        logger.info(f"No results for query '{query_text}' (attempt {attempt} of {limit}). Retrying after delay...")
                        time.sleep(10)

            if not results:
                logger.warning(f"Incomplete retrieval for query '{query_text}' in dataset_id {dataset_id}.")
                complete_success = False
            search_outputs.append(results)

        if complete_success:
            ground_truth_df["search_output"] = search_outputs
            try:
                ground_truth_df.to_csv(output_csv_path, index=False)
                logger.info(f"Saved search outputs for dataset_id {dataset_id} to {output_csv_path}")
                summary[dataset_id] = "Success"
            except Exception as e:
                logger.error(f"Error saving search outputs for dataset_id {dataset_id}: {e}")
                summary[dataset_id] = f"Error saving outputs: {e}"
        else:
            logger.warning(f"Incomplete retrieval for dataset_id {dataset_id}; Increase limit and retry.")
            summary[dataset_id] = "Incomplete Retrieval; Needs Attention"

    logger.info("Retrieval Stage Summary:")
    for ds_id, result in summary.items():
        logger.info(f"Dataset {ds_id}: {result}")
def rag_retrieval() -> None:
    """
    Main entry point:
      1. Load default config from YAML (config.yaml).
      2. Merge CLI overrides using OmegaConf.from_cli().
      3. Process Excel file (optionally limit rows).
      4. Extract metadata with get_meta_data.
      5. Initialize SparkCli.
      6. Execute pipeline stage: upload, delete, status, or retrieval.
    
    Note: No docid_path usage anywhere.
    """
    logger.info("Running rag_retrieval pipeline")

    # 2. Load default config, then merge any CLI overrides
    try:
        config_path = "/home/jovyan/llm-leaderboard-evaluation/src/llm_leaderboard_evaluation/pipeline.yaml"
        default_conf = OmegaConf.load(config_path)
    except FileNotFoundError:
        logger.error(f"{config_path} not found. Exiting.")
        return
    except Exception as exc:
        logger.exception(f"Error loading {config_path}: {exc}")
        return

    cli_conf = OmegaConf.from_cli()
    cfg: DictConfig = OmegaConf.merge(default_conf, cli_conf)
    logger.info(f"Final merged config:\n{OmegaConf.to_yaml(cfg, resolve=True)}")

    # 3. Extract relevant config pieces
    preprocess_cfg = cfg.get("preprocess", {})
    pipeline_cfg = cfg.get("pipeline", {})
    search_cfg = cfg.get("search", {})
    row_selection = pipeline_cfg.get("row_selection", -1)

    # 4. Process Excel file (optionally limit rows)
    try:
        df = process_csv_file(
            preprocess_cfg=preprocess_cfg,
            row_selection=row_selection
        )
    except Exception as exc:
        logger.error(f"Excel processing failed: {exc}")
        return

    # 4. Extract metadata from the processed DataFrame
    try:
        meta_data_list = get_meta_data(df)
    except Exception as exc:
        logger.error(f"Metadata extraction failed: {exc}")
        return

    # 5. Initialize SparkCli
    try:
        cli = SparkCli()
        logger.info("SparkCli client initialized.")
    except Exception as exc:
        logger.exception("Failed to initialize SparkCli client.")
        return

    # 6. Execute pipeline stage
    stage = pipeline_cfg.get("stage", None)
    logger.info(f"Pipeline stage: {stage}")

    workflow_successful = True  # Flag to track workflow success
    try:
        if stage == "upload":
            upload_cfg = pipeline_cfg.get("upload", {})
            try:
                dataset_upload(meta_data_list=meta_data_list, cli=cli, upload_cfg=upload_cfg)
            except Exception as e:
                logger.exception("Upload failed")
                workflow_successful = False
                raise e
            else:
                logger.info("Upload completed successfully")
                print("Upload completed successfully")
        elif stage == "delete":
            delete_cfg = pipeline_cfg.get("delete", {})  # This config is not used, as mentioned
            try:
                dataset_delete(cli=cli, df=df)
            except Exception as e:
                logger.exception("Deletion failed")
                workflow_successful = False
                raise e
            else:
                logger.info("Deletion completed successfully")
                print("Deletion completed successfully")
        elif stage == "status":
            status_cfg = pipeline_cfg.get("status", {})
            try:
                dataset_ensure_ready(cli=cli, meta_data_list=meta_data_list, status_cfg=status_cfg)
            except Exception as e:
                logger.exception("Status check failed")
                workflow_successful = False
                raise e
            else:
                logger.info("Status check completed successfully")
                print("Status check completed successfully")
        elif stage == "retrieval":
            try:
                retrieval_cfg = pipeline_cfg.get("retrieval", {})
                dataset_retrieval(cli=cli, df=df, retrieval_cfg=retrieval_cfg)
            except Exception as e:
                logger.exception("rag_retrieval failed")
                workflow_successful = False
                raise e
            else:
                logger.info("rag_retrieval completed successfully")
                print("rag_retrieval completed successfully")
        elif stage is None:
            logger.warning("Stage not set. Please set the stage using: python pipeline_rag_retrieval.py pipeline.stage=<stage>")
            workflow_successful = False
        else:
            logger.warning(f"Unknown pipeline stage: {stage}. Choose from 'upload', 'delete', 'status', 'retrieval'")
            workflow_successful = False
    except Exception:
        workflow_successful = False
    finally:
        if workflow_successful:
            logger.info("=== Main workflow completed ===")
        else:
            logger.info("=== Main workflow failed ===")
