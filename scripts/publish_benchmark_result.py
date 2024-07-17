# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
from datetime import datetime
import json
import os
import pathlib
from pathlib import Path
import subprocess
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd
import pandas_gbq

LOGGING_NAME_ENV_VAR = "BIGFRAMES_PERFORMANCE_LOG_NAME"
CURRENT_DIRECTORY = pathlib.Path(__file__).parent.absolute()


def run_benchmark(base_path):
    # Run benchmarks in parallel session.run's, since each benchmark
    # takes an environment variable for performance logging
    processes = process_benchmark_recursively(base_path)
    for process in processes:
        process.wait()
    return collect_benchmark_result(base_path)


def run_notebook_benchmark():
    notebooks_list = list(Path("notebooks/").glob("*/*.ipynb"))

    denylist = [
        # Regionalized testing is manually added later.
        "notebooks/location/regionalized.ipynb",
        # These notebooks contain special colab `param {type:"string"}`
        # comments, which make it easy for customers to fill in their
        # own information.
        #
        # With the notebooks_fill_params.py script, we are able to find and
        # replace the PROJECT_ID parameter, but not the others.
        #
        # TODO(ashleyxu): Test these notebooks by replacing parameters with
        # appropriate values and omitting cleanup logic that may break
        # our test infrastructure.
        "notebooks/getting_started/ml_fundamentals_bq_dataframes.ipynb",  # Needs DATASET.
        "notebooks/regression/bq_dataframes_ml_linear_regression.ipynb",  # Needs DATASET_ID.
        "notebooks/generative_ai/bq_dataframes_ml_drug_name_generation.ipynb",  # Needs CONNECTION.
        # TODO(b/332737009): investigate why we get 404 errors, even though
        # bq_dataframes_llm_code_generation creates a bucket in the sample.
        "notebooks/generative_ai/bq_dataframes_llm_code_generation.ipynb",  # Needs BUCKET_URI.
        "notebooks/generative_ai/sentiment_analysis.ipynb",  # Too slow
        "notebooks/vertex_sdk/sdk2_bigframes_pytorch.ipynb",  # Needs BUCKET_URI.
        "notebooks/vertex_sdk/sdk2_bigframes_sklearn.ipynb",  # Needs BUCKET_URI.
        "notebooks/vertex_sdk/sdk2_bigframes_tensorflow.ipynb",  # Needs BUCKET_URI.
        # The experimental notebooks imagine features that don't yet
        # exist or only exist as temporary prototypes.
        "notebooks/experimental/longer_ml_demo.ipynb",
        # The notebooks that are added for more use cases, such as backing a
        # blog post, which may take longer to execute and need not be
        # continuously tested.
        "notebooks/apps/synthetic_data_generation.ipynb",
    ]

    # Convert each Path notebook object to a string using a list comprehension.
    notebooks = [str(nb) for nb in notebooks_list]

    # Remove tests that we choose not to test.
    notebooks = list(filter(lambda nb: nb not in denylist, notebooks))

    # Regionalized notebooks
    notebooks_reg = {
        "regionalized.ipynb": [
            "asia-southeast1",
            "eu",
            "europe-west4",
            "southamerica-west1",
            "us",
            "us-central1",
        ]
    }
    notebooks_reg = {
        os.path.join("notebooks/location", nb): regions
        for nb, regions in notebooks_reg.items()
    }

    # The pytest --nbmake exits silently with "no tests ran" message if
    # one of the notebook paths supplied does not exist. Let's make sure that
    # each path exists.
    for nb in notebooks + list(notebooks_reg):
        assert os.path.exists(nb), nb

    # TODO(shobs): For some reason --retries arg masks exceptions occurred in
    # notebook failures, and shows unhelpful INTERNALERROR. Investigate that
    # and enable retries if we can find a way to surface the real exception
    # bacause the notebook is running against real GCP and something may fail
    # due to transient issues.
    pytest_command = [
        "py.test",
        "--nbmake",
        "--nbmake-timeout=900",  # 15 minutes
    ]

    try:
        # Populate notebook parameters and make a backup so that the notebooks
        # are runnable.
        notebooks_fill_params(notebooks)

        # Run notebooks in parallel session.run's, since each notebook
        # takes an environment variable for performance logging
        processes = []
        for notebook in notebooks:
            process = benchmark_process(
                args=(*pytest_command, notebook),
                log_env_name_var=os.path.basename(notebook),
                filename=notebook,
            )
            processes.append(process)

        for process in processes:
            process.wait()

    finally:
        # Prevent our notebook changes from getting checked in to git
        # accidentally.
        notebooks_restore_from_backup(notebooks)

    # Additionally run regionalized notebooks in parallel session.run's.
    # Each notebook takes a different region via env param.
    processes = []
    for notebook, regions in notebooks_reg.items():
        for region in regions:
            notebook_benchmark_path = f"{notebook}_{region}"
            process = benchmark_process(
                args=(*pytest_command, notebook),
                log_env_name_var=os.path.basename(notebook_benchmark_path),
                filename=notebook_benchmark_path,
                region=region,
            )
            processes.append(process)

    for process in processes:
        process.wait()

    # when the environment variable is set as it is above,
    # notebooks output a .bytesprocessed and .slotmillis report
    # collect those reports and print a summary
    return collect_benchmark_result(Path("notebooks/"))


def process_benchmark_recursively(
    current_path: Path,
    benchmark_configs: List[Tuple[Optional[str], List[str]]] = [(None, [])],
):
    config_path = current_path / "config.jsonl"
    if config_path.exists():
        benchmark_configs = []
        with open(config_path, "r") as f:
            for line in f:
                config = json.loads(line)
                python_args = [f"--{key}={value}" for key, value in config.items()]
                suffix = (
                    config["benchmark_suffix"]
                    if "benchmark_suffix" in config
                    else "_".join(f"{key}_{value}" for key, value in config.items())
                )
                benchmark_configs.append((suffix, python_args))

    benchmark_script_list = list(current_path.glob("*.py"))
    processes = []
    for benchmark in benchmark_script_list:
        if benchmark.name == "utils.py":
            continue
        for benchmark_config in benchmark_configs:
            args = ["python", str(benchmark)]
            args.extend(benchmark_config[1])
            log_env_name_var = str(benchmark)
            if benchmark_config[0] is not None:
                log_env_name_var += f"_{benchmark_config[0]}"
            process = benchmark_process(args=args, log_env_name_var=log_env_name_var)
            processes.append(process)

    for sub_dir in [d for d in current_path.iterdir() if d.is_dir()]:
        processes.extend(process_benchmark_recursively(sub_dir, benchmark_configs))
    return processes


def notebooks_fill_params(notebooks):
    command = (
        "python",
        CURRENT_DIRECTORY / "notebooks_fill_params.py",
        *notebooks,
    )
    subprocess.run(command, env=os.environ, check=True)


def notebooks_restore_from_backup(notebooks):
    command = (
        "python",
        CURRENT_DIRECTORY / "notebooks_restore_from_backup.py",
        *notebooks,
    )
    subprocess.run(command, env=os.environ, check=True)


def benchmark_process(args, log_env_name_var, filename=None, region=None):
    env = os.environ.copy()
    if region:
        env["BIGQUERY_LOCATION"] = region
    env[LOGGING_NAME_ENV_VAR] = log_env_name_var
    process = subprocess.Popen(args, env=env)
    return process


def collect_benchmark_result(path: Path) -> pd.DataFrame:
    """Generate a DataFrame report on HTTP queries, bytes processed, slot time and execution time from log files."""
    try:
        results_dict: Dict[str, List[Union[int, float, None]]] = {}
        bytes_files = sorted(path.rglob("*.bytesprocessed"))
        millis_files = sorted(path.rglob("*.slotmillis"))
        bq_seconds_files = sorted(path.rglob("*.bq_exec_time_seconds"))

        local_seconds_files = sorted(path.rglob("*.local_exec_time_seconds"))
        has_local_seconds = len(local_seconds_files) > 0

        if has_local_seconds:
            if not (
                len(bytes_files)
                == len(millis_files)
                == len(local_seconds_files)
                == len(bq_seconds_files)
            ):
                raise ValueError(
                    "Mismatch in the number of report files for bytes, millis, and seconds."
                )
        else:
            if not (len(bytes_files) == len(millis_files) == len(bq_seconds_files)):
                raise ValueError(
                    "Mismatch in the number of report files for bytes, millis, and seconds."
                )

        for idx in range(len(bytes_files)):
            bytes_file = bytes_files[idx]
            millis_file = millis_files[idx]
            bq_seconds_file = bq_seconds_files[idx]
            filename = bytes_file.relative_to(path).with_suffix("")

            if filename != millis_file.relative_to(path).with_suffix(
                ""
            ) or filename != bq_seconds_file.relative_to(path).with_suffix(""):
                raise ValueError(
                    "File name mismatch among bytes, millis, and seconds reports."
                )

            if has_local_seconds:
                local_seconds_file = local_seconds_files[idx]
                if filename != local_seconds_file.relative_to(path).with_suffix(""):
                    raise ValueError(
                        "File name mismatch among bytes, millis, and seconds reports."
                    )

            with open(bytes_file, "r") as file:
                lines = file.read().splitlines()
                query_count = len(lines)
                total_bytes = sum(int(line) for line in lines)

            with open(millis_file, "r") as file:
                lines = file.read().splitlines()
                total_slot_millis = sum(int(line) for line in lines)

            if has_local_seconds:
                with open(local_seconds_file, "r") as file:
                    local_seconds = float(file.readline().strip())
            else:
                local_seconds = None

            with open(bq_seconds_file, "r") as file:
                lines = file.read().splitlines()
                bq_seconds = sum(float(line) for line in lines)

            results_dict[str(filename)] = [
                query_count,
                total_bytes,
                total_slot_millis,
                local_seconds,
                bq_seconds,
            ]
    finally:
        for files_to_remove in (
            path.rglob("*.bytesprocessed"),
            path.rglob("*.slotmillis"),
            path.rglob("*.local_exec_time_seconds"),
            path.rglob("*.bq_exec_time_seconds"),
        ):
            for log_file in files_to_remove:
                log_file.unlink()

    columns = [
        "Query_Count",
        "Bytes_Processed",
        "Slot_Millis",
        "Local_Execution_Time_Sec",
        "BigQuery_Execution_Time_Sec",
    ]

    benchmark_metrics = pd.DataFrame.from_dict(
        results_dict,
        orient="index",
        columns=columns,
    )

    print("---BIGQUERY USAGE REPORT---")
    for index, row in benchmark_metrics.iterrows():
        print(
            f"{index} - query count: {row['Query_Count']},"
            f" bytes processed sum: {row['Bytes_Processed']},"
            f" slot millis sum: {row['Slot_Millis']},"
            f" local execution time: {row['Local_Execution_Time_Sec']} seconds"
            f" bigquery execution time: {row['BigQuery_Execution_Time_Sec']} seconds"
        )

    cumulative_queries = benchmark_metrics["Query_Count"].sum()
    cumulative_bytes = benchmark_metrics["Bytes_Processed"].sum()
    cumulative_slot_millis = benchmark_metrics["Slot_Millis"].sum()
    cumulative_local_seconds = benchmark_metrics["Local_Execution_Time_Sec"].sum(
        skipna=True
    )
    cumulative_bq_seconds = benchmark_metrics["BigQuery_Execution_Time_Sec"].sum()

    print(
        f"---total queries: {cumulative_queries}, "
        f"total bytes: {cumulative_bytes}, "
        f"total slot millis: {cumulative_slot_millis}---"
        f"Total local execution time: {cumulative_local_seconds} seconds---"
        f"Total bigquery execution time: {cumulative_bq_seconds} seconds---"
    )

    return benchmark_metrics


def get_repository_status():
    """
    Retrieves the current timestamp, and whether it is run by kokoro.
    """
    return {
        "benchmark_start_time": datetime.now().isoformat(),
        "is_running_in_kokoro": "KOKORO_JOB_NAME" in os.environ,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Run benchmarks for different scenarios."
    )
    parser.add_argument(
        "--notebook",
        action="store_true",
        help="Run benchmarks associated with notebooks",
    )
    args = parser.parse_args()

    repo_status = get_repository_status()
    if args.notebook:
        bigquery_table = "bigframes-metrics.benchmark_report.notebook_benchmark"
        benchmark_metrics = run_notebook_benchmark()
    else:
        bigquery_table = "bigframes-metrics.benchmark_report.benchmark"
        benchmark_metrics = run_benchmark(Path("tests/benchmark/"))

    for idx, col in enumerate(repo_status.keys()):
        benchmark_metrics.insert(idx, col, repo_status[col])

    pandas_gbq.to_gbq(
        dataframe=benchmark_metrics,
        destination_table=bigquery_table,
        if_exists="append",
    )


if __name__ == "__main__":
    main()
