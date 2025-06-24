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


def generate_random_data(schema: list[tuple[str, str, int | None]],
                         num_rows: int,
                         rng: np.random.Generator,
                         batch_size: int) -> iter:
    """
    Generates random data for the given schema and number of rows, yielding batches.
    """
    if num_rows == 0:
        yield []
        return

    char_list = list("abcdefghijklmnopqrstuvwxyz0123456789")
    json_char_list = list("abcdef")
    col_names_ordered = [s[0] for s in schema]

    generated_rows_total = 0
    while generated_rows_total < num_rows:
        current_batch_size = min(batch_size, num_rows - generated_rows_total)
        if current_batch_size == 0:
            break

        columns_data_batch = {}
        for col_name, bq_type, length in schema:
            if bq_type == 'BOOL':
                columns_data_batch[col_name] = rng.choice([True, False], size=current_batch_size)
            elif bq_type == 'INT64':
                columns_data_batch[col_name] = rng.integers(-10**18, 10**18, size=current_batch_size, dtype=np.int64)
            elif bq_type == 'FLOAT64':
                columns_data_batch[col_name] = rng.random(size=current_batch_size) * 2 * 10**10 - 10**10
            elif bq_type == 'NUMERIC':
                raw_numerics = rng.random(size=current_batch_size) * 2 * 10**28 - 10**28
                format_numeric_v = np.vectorize(lambda x: f"{x:.9f}")
                columns_data_batch[col_name] = format_numeric_v(raw_numerics)
            elif bq_type == 'DATE':
                start_date_ord = datetime.date(1, 1, 1).toordinal()
                max_days = (datetime.date(9999, 12, 31) - datetime.date(1, 1, 1)).days
                day_offsets = rng.integers(0, max_days + 1, size=current_batch_size)
                date_ordinals = start_date_ord + day_offsets
                columns_data_batch[col_name] = [datetime.date.fromordinal(int(ordinal)).isoformat() for ordinal in date_ordinals]
            elif bq_type == 'DATETIME':
                years = rng.integers(1, 10000, size=current_batch_size)
                months = rng.integers(1, 13, size=current_batch_size)
                days_val = rng.integers(1, 29, size=current_batch_size) # Simplified day generation
                hours = rng.integers(0, 24, size=current_batch_size)
                minutes = rng.integers(0, 60, size=current_batch_size)
                seconds = rng.integers(0, 60, size=current_batch_size)
                microseconds = rng.integers(0, 1000000, size=current_batch_size)

                # Construct Python datetime objects then convert to numpy.datetime64 for string conversion
                py_datetimes = []
                for i in range(current_batch_size):
                    try:
                        py_datetimes.append(datetime.datetime(
                            years[i], months[i], days_val[i],
                            hours[i], minutes[i], seconds[i], microseconds[i]))
                    except ValueError: # Fallback for invalid date component combinations
                        py_datetimes.append(datetime.datetime(
                            2000, 1, 1, hours[i], minutes[i], seconds[i], microseconds[i]))

                np_datetimes = np.array(py_datetimes, dtype='datetime64[us]')
                # np.datetime_as_string produces 'YYYY-MM-DDTHH:MM:SS.ffffff'
                # BQ DATETIME typically uses a space separator: 'YYYY-MM-DD HH:MM:SS.ffffff'
                datetime_strings = np.datetime_as_string(np_datetimes, unit='us')
                columns_data_batch[col_name] = [s.replace('T', ' ') for s in datetime_strings]

            elif bq_type == 'TIMESTAMP':
                # Generate seconds from a broad range (e.g., year 1 to 9999)
                # Note: Python's datetime.timestamp() might be limited by system's C mktime.
                # For broader range with np.datetime64, it's usually fine.
                # Let's generate epoch seconds relative to Unix epoch for np.datetime64 compatibility
                min_epoch_seconds = int(datetime.datetime(1, 1, 1, 0, 0, 0, tzinfo=datetime.timezone.utc).timestamp())
                # Max for datetime64[s] is far out, but let's bound it reasonably for BQ.
                max_epoch_seconds = int(datetime.datetime(9999, 12, 28, 23, 59, 59, tzinfo=datetime.timezone.utc).timestamp())

                epoch_seconds = rng.integers(min_epoch_seconds, max_epoch_seconds + 1, size=current_batch_size, dtype=np.int64)
                microseconds_offset = rng.integers(0, 1000000, size=current_batch_size, dtype=np.int64)

                # Create datetime64[s] from epoch seconds and add microseconds as timedelta64[us]
                np_timestamps_s = epoch_seconds.astype('datetime64[s]')
                np_microseconds_td = microseconds_offset.astype('timedelta64[us]')
                np_timestamps_us = np_timestamps_s + np_microseconds_td

                # Convert to string with UTC timezone indicator
                # np.datetime_as_string with timezone='UTC' produces 'YYYY-MM-DDTHH:MM:SS.ffffffZ'
                # BigQuery generally accepts this for TIMESTAMP.
                columns_data_batch[col_name] = np.datetime_as_string(np_timestamps_us, unit='us', timezone='UTC')

            elif bq_type == 'TIME':
                hours = rng.integers(0, 24, size=current_batch_size)
                minutes = rng.integers(0, 60, size=current_batch_size)
                seconds = rng.integers(0, 60, size=current_batch_size)
                microseconds = rng.integers(0, 1000000, size=current_batch_size)
                time_list = []
                for i in range(current_batch_size):
                    time_list.append(datetime.time(
                        hours[i], minutes[i], seconds[i], microseconds[i]
                    ).isoformat())
                columns_data_batch[col_name] = time_list
            elif bq_type == 'STRING':
                content_len = length if length is not None else 1
                content_len = max(0, content_len)
                if content_len == 0:
                    columns_data_batch[col_name] = [""] * current_batch_size
                else:
                    chars_array = rng.choice(char_list, size=(current_batch_size, content_len))
                    columns_data_batch[col_name] = [''.join(row_chars) for row_chars in chars_array]
            elif bq_type == 'BYTES':
                content_len = length if length is not None else 1
                content_len = max(0, content_len)
                columns_data_batch[col_name] = [rng.bytes(content_len) for _ in range(current_batch_size)]
            elif bq_type == 'JSON':
                content_len = length if length is not None else 10
                json_list = []
                for _ in range(current_batch_size):
                    if content_len <= 5:
                        json_val_len = max(0, content_len - 5)
                        json_val_chars = rng.choice(json_char_list, size=json_val_len)
                        json_obj = {"k": ''.join(json_val_chars)} if content_len > 4 else ""
                    else:
                        val_len = max(1, content_len - 10)
                        json_val_chars = rng.choice(json_char_list, size=val_len)
                        json_obj = {"key": ''.join(json_val_chars)}
                    json_str = json.dumps(json_obj)
                    if len(json_str.encode('utf-8')) > content_len and content_len > 0:
                        approx_val_len = max(1, content_len - len(json.dumps({"key":""}).encode('utf-8')))
                        json_obj_adjusted = {"key": "X" * approx_val_len}
                        json_str = json.dumps(json_obj_adjusted)
                        json_str = json_str[:content_len]
                    elif len(json_str.encode('utf-8')) < content_len and content_len > 0:
                        if "key" in json_obj and isinstance(json_obj["key"], str):
                            padding_needed = content_len - len(json_str.encode('utf-8'))
                            json_obj["key"] += "X" * padding_needed
                            json_str = json.dumps(json_obj)
                            while len(json_str.encode('utf-8')) > content_len and len(json_obj["key"]) > 0:
                                json_obj["key"] = json_obj["key"][:-1]
                                json_str = json.dumps(json_obj)
                            if len(json_str.encode('utf-8')) > content_len:
                                 json_str = json_str[:content_len]
                    json_list.append(json_str)
                columns_data_batch[col_name] = json_list

        # Assemble batch of rows
        batch_data = []
        if current_batch_size > 0:
            for i in range(current_batch_size):
                row = {col_name: columns_data_batch[col_name][i] for col_name in col_names_ordered}
                batch_data.append(row)

        yield batch_data
        generated_rows_total += current_batch_size


def create_and_load_table(
    project_id: str,
    dataset_id: str,
    table_name: str,
    schema_def: list[tuple[str, str, int | None]],
    num_rows: int,
    rng: np.random.Generator,
    data_gen_batch_size: int  # Batch size for the data generator
):
    """Creates a BigQuery table and loads data into it by consuming a data generator."""

    # BQ client library streaming insert batch size (rows per API call)
    # This is different from data_gen_batch_size which is for generating data.
    # We can make BQ_LOAD_BATCH_SIZE smaller than data_gen_batch_size if needed.
    BQ_LOAD_BATCH_SIZE = 500

    if not project_id or project_id == "your-gcp-project":
        print(f"SKIPPING BigQuery interaction for table {table_name}: PROJECT_ID not set.")
        print(f"  Schema: {schema_def}")
        print(f"  Total rows to generate: {num_rows}")
        print(f"  Data generation batch size: {data_gen_batch_size}")

        # Simulate consuming the first batch for sample output
        first_batch_generated = False
        for i, batch_data in enumerate(generate_random_data(schema_def, num_rows, rng, data_gen_batch_size)):
            if not first_batch_generated and batch_data:
                print(f"  Simulating: Would load {len(batch_data)} row(s) in first batch (sample): {batch_data[0]}")
                first_batch_generated = True
            elif not first_batch_generated and not batch_data and num_rows == 0:
                 print(f"  Simulating: Would load 0 rows as num_rows is 0.")
                 first_batch_generated = True # Mark as handled

            if i == 0: # Only show info for the first yielded batch in simulation
                if num_rows > 0 and not batch_data: # Should not happen if num_rows > 0
                    print("  Simulating: Generator yielded an empty first batch unexpectedly for non-zero num_rows.")
                elif num_rows == 0 and not batch_data: # Expected for num_rows = 0
                    pass # Already handled
                # If num_rows > 0 and batch_data has content, it's handled above.

            if i == 0 and num_rows > data_gen_batch_size: # If there will be more batches
                num_simulated_batches = math.ceil(num_rows / data_gen_batch_size)
                print(f"  Simulating: Would continue generating and loading in approx. {num_simulated_batches} batches.")
            elif i == 0 and num_rows > 0 and num_rows <= data_gen_batch_size : # Single batch
                 print(f"  Simulating: All {num_rows} rows would be generated in a single batch.")

            if i == 0: # Break after inspecting the first batch for simulation purposes
                break
        if not first_batch_generated and num_rows > 0 : # If loop didn't run (e.g. num_rows was 0 initially for generator)
             print(f"  Simulating: No data batches were generated (num_rows: {num_rows}).")
        return

    client = bigquery.Client(project=project_id)
    table_id = f"{project_id}.{dataset_id}.{table_name}"

    bq_schema = []
    for col_name, type_name, _ in schema_def:
        bq_schema.append(bigquery.SchemaField(col_name, type_name))

    table = bigquery.Table(table_id, schema=bq_schema)

    try:
        print(f"Creating table {table_id}...")
        client.create_table(table, exists_ok=True)
        print(f"Table {table_id} created successfully or already exists.")

        if num_rows > 0:
            print(f"Starting to load {num_rows} rows into {table_id} in batches...")
            total_rows_loaded = 0

            # Data is generated in data_gen_batch_size chunks by the generator.
            # Data is loaded into BQ in BQ_LOAD_BATCH_SIZE chunks.
            # These can be different. The generator yields larger chunks,
            # and we further batch them for BQ API calls if needed.

            batch_num_gen = 0
            for generated_data_chunk in generate_random_data(schema_def, num_rows, rng, data_gen_batch_size):
                batch_num_gen += 1
                if not generated_data_chunk: # Should only happen if num_rows was 0 from start
                    continue

                print(f"  Processing generated chunk {batch_num_gen} (size: {len(generated_data_chunk)} rows)...")

                # Sub-batch for BigQuery client.insert_rows_json
                for i in range(0, len(generated_data_chunk), BQ_LOAD_BATCH_SIZE):
                    bq_insert_batch = generated_data_chunk[i:i + BQ_LOAD_BATCH_SIZE]
                    if not bq_insert_batch:
                        continue

                    errors = client.insert_rows_json(table, bq_insert_batch)
                    if errors:
                        print(f"    Encountered errors while inserting sub-batch: {errors}")
                        # TODO: Add more robust error handling, e.g., retry, log to file
                    else:
                        total_rows_loaded += len(bq_insert_batch)
                        print(f"    Loaded sub-batch of {len(bq_insert_batch)} rows. Total loaded: {total_rows_loaded}/{num_rows}")

            if total_rows_loaded == num_rows:
                print(f"Data loading complete for {table_id}. Total {total_rows_loaded} rows loaded.")
            else:
                print(f"Warning: Data loading for {table_id} finished, but row count mismatch. Loaded: {total_rows_loaded}, Expected: {num_rows}")
        else:
            print(f"No data to load for table {table_id} as num_rows is 0.")

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

        # Data generation batch size - can be tuned.
        # Larger might be more efficient for generation, but uses more memory per batch.
        # BQ loading itself is handled in smaller sub-batches within create_and_load_table.
        DATA_GENERATION_BATCH_SIZE = 10000

        # No longer pre-generate data_to_load here.
        # create_and_load_table will now pull from the generator.
        create_and_load_table(
            PROJECT_ID,
            DATASET_ID,
            table_name,
            schema_definition,
            num_rows, # Pass the total number of rows
            rng,      # Pass the random number generator
            DATA_GENERATION_BATCH_SIZE # Pass the data generation batch size
        )

if __name__ == "__main__":
    main()
