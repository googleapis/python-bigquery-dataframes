import io

import google.cloud.storage as storage  # type: ignore


def test_to_parquet_writes_file(
    scalars_df_index, gcs_folder: str, gcs_client: storage.Client
):
    path = gcs_folder + "test_to_parquet_writes_file.parquet"
    # TODO(swast): Do a bit more processing on the input DataFrame to ensure
    # the exported results are from the generated query, not just the source
    # table.
    scalars_df_index.to_parquet(path)
    with io.BytesIO() as file:
        gcs_client.download_blob_to_file(path, file)
        # TODO(swast): Load pandas DataFrame from exported parquet file to
        # check that it matches the expected output.
        assert len(file.getvalue()) > 0
