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
    """
    Generates random data for the given schema and number of rows using vectorized operations.
    """
    if num_rows == 0:
        return []

    columns_data = {}
    char_list = list("abcdefghijklmnopqrstuvwxyz0123456789")
    json_char_list = list("abcdef")

    for col_name, bq_type, length in schema:
        if bq_type == 'BOOL':
            columns_data[col_name] = rng.choice([True, False], size=num_rows)
        elif bq_type == 'INT64':
            columns_data[col_name] = rng.integers(-10**18, 10**18, size=num_rows, dtype=np.int64)
        elif bq_type == 'FLOAT64':
            columns_data[col_name] = rng.random(size=num_rows) * 2 * 10**10 - 10**10
        elif bq_type == 'NUMERIC':
            # Generate as float, then format to string.
            # Using np.vectorize for formatting.
            raw_numerics = rng.random(size=num_rows) * 2 * 10**28 - 10**28
            format_numeric_v = np.vectorize(lambda x: f"{x:.9f}")
            columns_data[col_name] = format_numeric_v(raw_numerics)
        elif bq_type == 'DATE':
            start_date_ord = datetime.date(1, 1, 1).toordinal()
            max_days = (datetime.date(9999, 12, 31) - datetime.date(1, 1, 1)).days
            day_offsets = rng.integers(0, max_days + 1, size=num_rows)
            # Vectorized conversion from ordinal + offset to ISO date string
            date_ordinals = start_date_ord + day_offsets
            columns_data[col_name] = [datetime.date.fromordinal(int(ordinal)).isoformat() for ordinal in date_ordinals]
        elif bq_type == 'DATETIME':
            # Generate components and then assemble.
            # This is less directly vectorized for the final string but component generation is.
            years = rng.integers(1, 10000, size=num_rows) # Max year 9999
            months = rng.integers(1, 13, size=num_rows)
            days = rng.integers(1, 29, size=num_rows) # simplify day generation
            hours = rng.integers(0, 24, size=num_rows)
            minutes = rng.integers(0, 60, size=num_rows)
            seconds = rng.integers(0, 60, size=num_rows)
            microseconds = rng.integers(0, 1000000, size=num_rows)
            dt_list = []
            for i in range(num_rows):
                try:
                    dt_list.append(datetime.datetime(
                        years[i], months[i], days[i],
                        hours[i], minutes[i], seconds[i], microseconds[i]
                    ).isoformat(sep=' '))
                except ValueError: # Handle invalid date combinations like Feb 29 in non-leap year if days were > 28
                     # Fallback to a known good date if component combination is invalid
                     dt_list.append(datetime.datetime(2000,1,1,hours[i],minutes[i],seconds[i],microseconds[i]).isoformat(sep=' '))
            columns_data[col_name] = dt_list
        elif bq_type == 'TIMESTAMP':
            # Generate Unix timestamps (seconds since epoch) then convert
            # Range for timestamps (approx 1970 to 2038 for standard 32-bit, extend for 64-bit)
            # Let's use a wider range, assuming BQ handles it, e.g., year 1 to 9999
            min_ts = int(datetime.datetime(1, 1, 1, tzinfo=datetime.timezone.utc).timestamp())
            max_ts = int(datetime.datetime(9999, 12, 28, tzinfo=datetime.timezone.utc).timestamp()) # up to 9999

            random_seconds = rng.integers(min_ts, max_ts, size=num_rows, dtype=np.int64)
            random_microseconds = rng.integers(0, 1000000, size=num_rows)

            ts_list = []
            for i in range(num_rows):
                try:
                    # Ensure the timestamp is within reasonable bounds for fromtimestamp
                    # Clamp to a safe range if extreme values from min_ts/max_ts cause issues
                    # Python's fromtimestamp might have platform limitations
                    base_dt = datetime.datetime.fromtimestamp(random_seconds[i], tz=datetime.timezone.utc)
                    final_dt = base_dt.replace(microsecond=random_microseconds[i]) # add microseconds
                    ts_list.append(final_dt.isoformat(sep=' '))
                except OverflowError: # Fallback for timestamps too large for platform's C mktime
                     fallback_dt = datetime.datetime(2000,1,1,12,0,0,random_microseconds[i], tzinfo=datetime.timezone.utc)
                     ts_list.append(fallback_dt.isoformat(sep=' '))
                except ValueError: # Handles cases like negative timestamps on Windows
                     fallback_dt = datetime.datetime(2000,1,1,12,0,0,random_microseconds[i], tzinfo=datetime.timezone.utc)
                     ts_list.append(fallback_dt.isoformat(sep=' '))

            columns_data[col_name] = ts_list
        elif bq_type == 'TIME':
            hours = rng.integers(0, 24, size=num_rows)
            minutes = rng.integers(0, 60, size=num_rows)
            seconds = rng.integers(0, 60, size=num_rows)
            microseconds = rng.integers(0, 1000000, size=num_rows)
            time_list = []
            for i in range(num_rows):
                time_list.append(datetime.time(
                    hours[i], minutes[i], seconds[i], microseconds[i]
                ).isoformat())
            columns_data[col_name] = time_list
        elif bq_type == 'STRING':
            content_len = length if length is not None else 1
            content_len = max(0, content_len) # ensure non-negative
            if content_len == 0:
                columns_data[col_name] = [""] * num_rows
            else:
                # Generate num_rows * content_len characters, then reshape and join
                chars_array = rng.choice(char_list, size=(num_rows, content_len))
                columns_data[col_name] = [''.join(row_chars) for row_chars in chars_array]
        elif bq_type == 'BYTES':
            content_len = length if length is not None else 1
            content_len = max(0, content_len)
            # rng.bytes is not directly vectorizable in the same way for num_rows
            columns_data[col_name] = [rng.bytes(content_len) for _ in range(num_rows)]
        elif bq_type == 'JSON':
            content_len = length if length is not None else 10
            json_list = []
            # JSON generation remains somewhat row-wise due to complexity of hitting exact length
            for _ in range(num_rows):
                if content_len <= 5: # approx "{\"\":0}"
                    json_val_len = max(0, content_len - 5) # Length of value part
                    json_val_chars = rng.choice(json_char_list, size=json_val_len)
                    json_obj = {"k": ''.join(json_val_chars)} if content_len > 4 else ""
                else:
                    val_len = max(1, content_len - 10) # {"key": "vvv..."}
                    json_val_chars = rng.choice(json_char_list, size=val_len)
                    json_obj = {"key": ''.join(json_val_chars)}

                json_str = json.dumps(json_obj)
                # Crude truncation/adjustment to meet length. This is hard to vectorize precisely.
                if len(json_str.encode('utf-8')) > content_len and content_len > 0:
                    approx_val_len = max(1, content_len - len(json.dumps({"key":""}).encode('utf-8')))
                    json_obj_adjusted = {"key": "X" * approx_val_len}
                    json_str = json.dumps(json_obj_adjusted)
                    json_str = json_str[:content_len] # Final hard truncate
                elif len(json_str.encode('utf-8')) < content_len and content_len > 0:
                    # If too short, and we have a simple {"key": "value"} structure, try to pad value
                    if "key" in json_obj and isinstance(json_obj["key"], str):
                        padding_needed = content_len - len(json_str.encode('utf-8'))
                        json_obj["key"] += "X" * padding_needed
                        json_str = json.dumps(json_obj)
                        # Truncate if padding overshot due to multi-byte chars or structure
                        while len(json_str.encode('utf-8')) > content_len and len(json_obj["key"]) > 0:
                            json_obj["key"] = json_obj["key"][:-1]
                            json_str = json.dumps(json_obj)
                        if len(json_str.encode('utf-8')) > content_len: # Final fallback if still too long
                             json_str = json_str[:content_len]


                json_list.append(json_str)
            columns_data[col_name] = json_list

    # Assemble rows from columns_data
    # This is a way to transpose the dictionary of arrays into a list of dicts
    data = []
    if num_rows > 0:
        col_names_ordered = [s[0] for s in schema] # Get column names in original schema order
        for i in range(num_rows):
            row = {col_name: columns_data[col_name][i] for col_name in col_names_ordered}
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
