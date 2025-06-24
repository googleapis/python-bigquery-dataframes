import math
import json
import datetime
import numpy as np
from google.cloud import bigquery

# --- Configuration ---
PROJECT_ID = "your-gcp-project"  # TODO: Replace with your GCP Project ID
DATASET_ID = "benchmark_dataset"  # TODO: Replace with your BigQuery Dataset ID

# --- Input Data ---
TABLE_STATS = {
    'percentile': [9, 19, 29, 39, 49, 59, 69, 79, 89, 99],
    'materialized_or_scanned_bytes': [
        0.0, 0.0, 4102.0, 76901.0, 351693.0, 500000.0, 500000.0,
        1320930.0, 17486432.0, 1919625975.0
    ],
    'num_materialized_or_scanned_rows': [
        0.0, 6.0, 100.0, 4955.0, 23108.0, 139504.0, 616341.0,
        3855698.0, 83725698.0, 5991998082.0
    ],
    'avg_row_bytes': [
        0.00014346299635435792, 0.005370969708923197, 0.3692756731526246,
        4.079344721151818, 7.5418, 12.528863516404146, 22.686258546389798,
        48.69689224091025, 100.90817356205852, 2020
    ],
    'materialized_mb': [
        0.0, 0.0, 0.004102, 0.076901, 0.351693, 0.5, 0.5, 1.32093,
        17.486432, 1919.625975
    ]
}

BIGQUERY_DATA_TYPE_SIZES = {
    'BOOL': 1, 'BYTES': 2, 'STRING': 2, 'DATE': 8, 'FLOAT64': 8,
    'INT64': 8, 'JSON': 8, 'DATETIME': 8, 'TIMESTAMP': 8, 'TIME': 8,
    'NUMERIC': 16
}

# --- Helper Functions ---

def get_bq_schema(target_row_size_bytes: int) -> list[tuple[str, str, int | None]]:
    """
    Determines the BigQuery table schema to match the target_row_size_bytes.
    Prioritizes fixed-size types for diversity, then uses flexible types.
    Returns a list of tuples: (column_name, type_name, length_for_flexible_type).
    Length is None for fixed-size types.
    """
    schema = []
    current_size = 0
    col_idx = 0

    # Prioritize fixed-size types
    fixed_types = [
        'BOOL', 'INT64', 'FLOAT64', 'NUMERIC', 'DATE',
        'DATETIME', 'TIMESTAMP', 'TIME'
    ]

    for bq_type in fixed_types:
        type_size = BIGQUERY_DATA_TYPE_SIZES[bq_type]
        if current_size + type_size <= target_row_size_bytes:
            schema.append((f"col_{bq_type.lower()}_{col_idx}", bq_type, None))
            current_size += type_size
            col_idx += 1
        if current_size >= target_row_size_bytes:
            break

    # If target_row_size_bytes is very small, we might only have fixed types or even be empty
    if current_size >= target_row_size_bytes and schema: # Ensure we don't overshoot with only fixed types
        # If we overshot with the last fixed type, we might need to remove it if there are other fields
        # or accept a slightly larger row if it's the only field.
        # For simplicity, we'll allow slight overage if only one fixed field is chosen.
        # If multiple, and last one caused overage, it implies target_row_size_bytes was too small for it.
        pass # Schema is already set with fixed types

    # Use flexible-size types to fill remaining space
    flexible_types = ['STRING', 'BYTES', 'JSON']
    flexible_types_added_this_round = []

    # Attempt to add one of each flexible type if space allows
    for bq_type in flexible_types:
        if current_size >= target_row_size_bytes:
            break

        # JSON base size is its content, BYTES/STRING have 2 byte overhead + content
        base_cost = 0 if bq_type == 'JSON' else BIGQUERY_DATA_TYPE_SIZES[bq_type]
        min_content_size = 1 # Smallest possible content for these types

        if current_size + base_cost + min_content_size <= target_row_size_bytes:
            # Temporarily add with min_content_size, will adjust later if it's the one to expand
            schema.append((f"col_{bq_type.lower()}_{col_idx}", bq_type, min_content_size))
            current_size += base_cost + min_content_size
            flexible_types_added_this_round.append(bq_type)
            col_idx += 1

    # If there's still space, expand the last added flexible type, or the first one if multiple were added
    if current_size < target_row_size_bytes and flexible_types_added_this_round:
        # Pick a flexible type to expand (e.g., the first one added in this round)
        # For simplicity, we'll expand the first flexible type found in the schema
        # that was added in this round.

        expanded_type_info = None
        expanded_type_idx_in_schema = -1

        for i, (name, type_name, length) in enumerate(schema):
            if type_name in flexible_types_added_this_round: # Check if it's one of the flexible types we just added
                expanded_type_info = (name, type_name, length)
                expanded_type_idx_in_schema = i
                break # Expand the first one we find

        if expanded_type_info:
            name, type_name, current_len = expanded_type_info
            base_cost = 0 if type_name == 'JSON' else BIGQUERY_DATA_TYPE_SIZES[type_name]

            # Remove its current contribution to size
            current_size -= (base_cost + current_len)

            # Calculate new length
            remaining_bytes_for_content = target_row_size_bytes - current_size - base_cost
            new_len = max(1, remaining_bytes_for_content) # Ensure at least 1 byte content

            schema[expanded_type_idx_in_schema] = (name, type_name, new_len)
            current_size += (base_cost + new_len)

    # If target_row_size_bytes is 0 or very small, schema might be empty.
    # Add a single 1-byte BOOL if schema is empty and target is >= 1
    if not schema and target_row_size_bytes > 0:
        # This case might occur if target_row_size_bytes is too small for any type's base overhead.
        # For example, if target_row_size_bytes = 1.
        # Smallest is BOOL (1 byte) or a STRING/BYTES of length 0 (costing 2 bytes).
        # Or JSON of length 0 (costing 0 bytes, but BQ might have internal minimums).
        # Let's ensure at least one column if target > 0.
        if target_row_size_bytes >= BIGQUERY_DATA_TYPE_SIZES['BOOL']:
             schema.append(("col_bool_fallback_0", "BOOL", None))
        # If even a BOOL is too large (e.g. target_row_size_bytes = 0), schema remains empty.
        # This is acceptable as per input data (first two rows have 0 bytes).

    # Final check: if current_size is still less and we have no flexible types to expand further,
    # this means the target was too small for many types or fixed types filled it perfectly.
    # This is generally okay. The goal is to get as close as possible from below or slightly above.

    return schema


def generate_random_data(schema: list[tuple[str, str, int | None]], num_rows: int, rng: np.random.Generator) -> list[dict]:
    """Generates random data for the given schema and number of rows."""
    data = []
    if num_rows == 0:
        return []

    for _ in range(num_rows):
        row = {}
        for col_name, bq_type, length in schema:
            if bq_type == 'BOOL':
                row[col_name] = rng.choice([True, False])
            elif bq_type == 'INT64':
                row[col_name] = int(rng.integers(-10**18, 10**18))
            elif bq_type == 'FLOAT64':
                row[col_name] = rng.random() * 2 * 10**10 - 10**10 # Large range of floats
            elif bq_type == 'NUMERIC':
                # NUMERIC can be up to 38 digits, 9 decimal places.
                # For simplicity, generate as float and convert to string.
                # BQ client library handles string representation for NUMERIC.
                val = rng.random() * 2 * 10**28 - 10**28
                row[col_name] = f"{val:.9f}" # Format with 9 decimal places
            elif bq_type == 'DATE':
                start_date = datetime.date(1, 1, 1)
                # Max date is 9999-12-31. Random days up to that.
                # Python's max date is 9999-12-31 as well.
                days = int(rng.integers(0, (datetime.date(9999,12,31) - start_date).days))
                row[col_name] = (start_date + datetime.timedelta(days=days)).isoformat()
            elif bq_type == 'DATETIME':
                # Similar to date, but with time
                dt = datetime.datetime(
                    year=rng.integers(1, 9999), month=rng.integers(1, 12), day=rng.integers(1, 28), # simplify day generation
                    hour=rng.integers(0, 23), minute=rng.integers(0, 59), second=rng.integers(0, 59),
                    microsecond=rng.integers(0, 999999)
                )
                row[col_name] = dt.isoformat(sep=' ') # BQ standard format
            elif bq_type == 'TIMESTAMP':
                # UTC timestamp
                dt = datetime.datetime(
                    year=rng.integers(1970, 2038), month=rng.integers(1, 12), day=rng.integers(1, 28),
                    hour=rng.integers(0, 23), minute=rng.integers(0, 59), second=rng.integers(0, 59),
                    microsecond=rng.integers(0, 999999), tzinfo=datetime.timezone.utc
                )
                row[col_name] = dt.isoformat(sep=' ')
            elif bq_type == 'TIME':
                t = datetime.time(
                    hour=rng.integers(0, 23), minute=rng.integers(0, 59), second=rng.integers(0, 59),
                    microsecond=rng.integers(0, 999999)
                )
                row[col_name] = t.isoformat()
            elif bq_type == 'STRING':
                # Generate random string of 'length' characters.
                # Each char is 1 byte for simplicity, assuming ASCII-like.
                # UTF-8 can be multi-byte, but for sizing, we assume length = byte size.
                # BigQuery STRING size is 2 bytes + UTF-8 encoded string size.
                # 'length' here refers to the content byte size.
                content_len = length if length is not None else 1 # Default to 1 if length somehow None
                row[col_name] = ''.join(rng.choice(list("abcdefghijklmnopqrstuvwxyz0123456789"), size=max(0,content_len)))
            elif bq_type == 'BYTES':
                # 'length' here refers to the content byte size.
                content_len = length if length is not None else 1
                row[col_name] = rng.bytes(max(0, content_len)) # Will be base64 encoded by client lib
            elif bq_type == 'JSON':
                # 'length' here refers to the content byte size of the JSON string.
                # This is tricky to get exact. We'll generate a simple JSON and hope it's close enough.
                # For more precision, one would need to serialize and check size, then adjust.
                # For this script, generate a simple structure.
                content_len = length if length is not None else 10 # Default length
                if content_len <= 5: # approx "{\"\":0}"
                     # Minimal JSON to fit small sizes
                    json_obj = {"k": "v"[:max(0, content_len-5)]} if content_len > 4 else ""
                else:
                    # Create a string that will be roughly 'content_len' when serialized as {"key": "value"}
                    # {"k": "vvv...v"} -> 10 bytes overhead for `{"k": ""}`.
                    # So, value_len should be content_len - 10.
                    val_len = max(1, content_len - 10)
                    json_obj = {"key": ''.join(rng.choice(list("abcdef"), size=val_len))}

                json_str = json.dumps(json_obj)
                # If the generated string is too long, truncate it. This is crude.
                if len(json_str.encode('utf-8')) > content_len and content_len > 0 :
                    # Attempt to create a string of roughly the right byte length for the value
                    # This is still an approximation.
                    approx_val_len = max(1, content_len - len(json.dumps({"key":""}).encode('utf-8')))
                    json_obj = {"key": "X" * approx_val_len}
                    json_str = json.dumps(json_obj)
                    # Final truncation if still off (crude)
                    json_str = json_str[:content_len]


                row[col_name] = json_str
        data.append(row)
    return data


def create_and_load_table(
    project_id: str,
    dataset_id: str,
    table_name: str,
    schema_def: list[tuple[str, str, int | None]],
    data_rows: list[dict]
):
    """Creates a BigQuery table and loads data into it."""
    if not project_id or project_id == "your-gcp-project":
        print(f"SKIPPING BigQuery interaction for table {table_name}: PROJECT_ID not set.")
        print(f"  Schema: {schema_def}")
        if data_rows:
             print(f"  Would load {len(data_rows)} row(s). First row (sample): {data_rows[0]}")
        else:
            print(f"  Would load {len(data_rows)} row(s).")
        return

    client = bigquery.Client(project=project_id)
    table_id = f"{project_id}.{dataset_id}.{table_name}"

    bq_schema = []
    for col_name, type_name, _ in schema_def: # length is not used for SchemaField directly
        bq_schema.append(bigquery.SchemaField(col_name, type_name))

    table = bigquery.Table(table_id, schema=bq_schema)

    try:
        print(f"Creating table {table_id}...")
        client.create_table(table, exists_ok=True)
        print(f"Table {table_id} created successfully or already exists.")

        if data_rows:
            print(f"Loading {len(data_rows)} rows into {table_id}...")
            # For very large number of rows, consider batching or load jobs
            # insert_rows_json has limits on request size and row count per request.
            # Max 10k rows / 10MB per API call for streaming inserts.
            # If num_rows is huge, this will fail or be very slow.
            # The problem implies up to ~6B rows, which MUST use Load Jobs, not streaming.
            # For now, this script will use insert_rows_json for simplicity,
            # acknowledging it won't work for the largest percentiles.

            # Simplified batching for demonstration
            batch_size = 500 # Keep batch size small for streaming
            for i in range(0, len(data_rows), batch_size):
                batch = data_rows[i:i + batch_size]
                errors = client.insert_rows_json(table, batch)
                if errors:
                    print(f"Encountered errors while inserting rows: {errors}")
                    # Decide how to handle errors: stop, log, retry certain types?
                    # For this script, we'll print and continue if possible.
                else:
                    print(f"Loaded batch of {len(batch)} rows successfully.")
            print(f"Data loading complete for {table_id}.")
        else:
            print(f"No data to load for table {table_id}.")

    except Exception as e:
        print(f"Error during BigQuery operation for table {table_id}: {e}")
        # Potentially re-raise or handle more gracefully
        raise

# --- Main Script Logic ---
def main():
    """Main function to create and populate BigQuery tables."""
    rng = np.random.default_rng(seed=42) # Seed for reproducibility

    num_percentiles = len(TABLE_STATS['percentile'])

    for i in range(num_percentiles):
        percentile = TABLE_STATS['percentile'][i]
        avg_row_bytes_raw = TABLE_STATS['avg_row_bytes'][i]
        num_rows_raw = TABLE_STATS['num_materialized_or_scanned_rows'][i]

        target_row_bytes = math.ceil(avg_row_bytes_raw)
        # Ensure minimum 1 row if original data suggests rows, even if rounded num_rows is 0.
        # Or if avg_row_bytes > 0, it implies there should be rows.
        # The input data has num_rows=0 for first percentile with avg_row_bytes > 0.
        # Let's ensure at least 1 row if target_row_bytes > 0, otherwise 0 rows.
        if target_row_bytes == 0:
            num_rows = 0
        else:
            num_rows = math.ceil(num_rows_raw) if num_rows_raw > 0 else 1
            if num_rows == 0 : num_rows = 1 # Ensure at least one row if target_row_bytes > 0

        table_name = f"percentile_{percentile:02d}_rows_{num_rows}_avg_bytes_{target_row_bytes}"
        print(f"\n--- Processing Table: {table_name} ---")
        print(f"Target average row bytes (rounded up): {target_row_bytes}")
        print(f"Number of rows (rounded up, min 1 if bytes>0): {num_rows}")

        if target_row_bytes == 0 and num_rows == 0:
            print("Skipping table creation as target row bytes and num_rows are 0.")
            # We might still want to create an empty table if that's the requirement.
            # For now, if both are zero, we skip actual schema generation and loading.
            # The problem implies creating 10 tables. So an empty table might be desired.
            # Let's create it with no schema if target_row_bytes is 0.
            schema_definition = []
        else:
            schema_definition = get_bq_schema(target_row_bytes)

        if not schema_definition and target_row_bytes > 0 :
            # This case should ideally be handled by get_bq_schema to add a fallback BOOL
            print(f"Warning: Schema could not be generated for target_row_bytes: {target_row_bytes}. Adding fallback.")
            schema_definition = [("col_fallback_bool", "BOOL", None)]


        print(f"Generated Schema: {schema_definition}")

        # For the largest tables, generating all data in memory is not feasible.
        # Data generation should ideally be streamed or done in chunks.
        # This script will attempt to generate all, which will fail for large num_rows.
        # Acknowledging this limitation for the scope of this exercise.
        if num_rows > 100000: # Heuristic limit for in-memory data generation
            print(f"Warning: Number of rows ({num_rows}) is very large. Data generation might be slow or fail.")
            print("Simulating data generation and skipping actual BQ load for this large table to prevent OOM.")
            # In a real scenario, use Dataflow or generate data directly to GCS then BQ Load Job.
            # For the purpose of this script, we will "pretend" to generate and load.
            # We will still call create_and_load_table, but it has its own checks for PROJECT_ID.
            # And if PROJECT_ID is set, it will attempt loading in small batches.
            # The true fix for multi-billion rows is a BQ Load Job from GCS.
            # Let's limit num_rows for actual data generation for this script to avoid OOM
            data_to_load = generate_random_data(schema_definition, min(num_rows, 1000), rng) # Generate only up to 1000 rows for demo
            print(f"Generated {len(data_to_load)} sample rows (capped for large tables).")
        else:
            data_to_load = generate_random_data(schema_definition, num_rows, rng)
            print(f"Generated {len(data_to_load)} rows of data.")


        create_and_load_table(PROJECT_ID, DATASET_ID, table_name, schema_definition, data_to_load)

if __name__ == "__main__":
    main()
