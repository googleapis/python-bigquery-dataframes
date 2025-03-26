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

from __future__ import annotations

import os
from typing import cast, Optional, Union
import warnings

import IPython.display as ipy_display
import pandas as pd
import requests

from bigframes import clients
from bigframes.core import log_adapter
import bigframes.dataframe
import bigframes.exceptions as bfe
from bigframes.operations import base
import bigframes.operations as ops
import bigframes.series

FILE_FOLDER_REGEX = r"^.*\/(.*)$"
FILE_EXT_REGEX = r"(\.[0-9a-zA-Z]+$)"


@log_adapter.class_logger
class BlobAccessor(base.SeriesMethods):
    """
    Blob functions for Series and Index.

    .. note::
        BigFrames Blob is still under experiments. It may not work and subject to change in the future.
    """

    def __init__(self, *args, **kwargs):
        if not bigframes.options.experiments.blob:
            raise NotImplementedError()

        super().__init__(*args, **kwargs)

    def uri(self) -> bigframes.series.Series:
        """URIs of the Blob.

        .. note::
            BigFrames Blob is still under experiments. It may not work and subject to change in the future.

        Returns:
            bigframes.series.Series: URIs as string."""
        s = bigframes.series.Series(self._block)

        return s.struct.field("uri")

    def authorizer(self) -> bigframes.series.Series:
        """Authorizers of the Blob.

        .. note::
            BigFrames Blob is still under experiments. It may not work and subject to change in the future.

        Returns:
            bigframes.series.Series: Autorithers(connection) as string."""
        s = bigframes.series.Series(self._block)

        return s.struct.field("authorizer")

    def version(self) -> bigframes.series.Series:
        """Versions of the Blob.

        .. note::
            BigFrames Blob is still under experiments. It may not work and subject to change in the future.

        Returns:
            bigframes.series.Series: Version as string."""
        # version must be retrieved after fetching metadata
        return self._apply_unary_op(ops.obj_fetch_metadata_op).struct.field("version")

    def metadata(self) -> bigframes.series.Series:
        """Retrieve the metadata of the Blob.

        .. note::
            BigFrames Blob is still under experiments. It may not work and subject to change in the future.

        Returns:
            bigframes.series.Series: JSON metadata of the Blob. Contains fields: content_type, md5_hash, size and updated(time)."""
        details_json = self._apply_unary_op(ops.obj_fetch_metadata_op).struct.field(
            "details"
        )
        import bigframes.bigquery as bbq

        return bbq.json_extract(details_json, "$.gcs_metadata").rename("metadata")

    def content_type(self) -> bigframes.series.Series:
        """Retrieve the content type of the Blob.

        .. note::
            BigFrames Blob is still under experiments. It may not work and subject to change in the future.

        Returns:
            bigframes.series.Series: string of the content type."""
        return (
            self.metadata()
            ._apply_unary_op(ops.JSONValue(json_path="$.content_type"))
            .rename("content_type")
        )

    def md5_hash(self) -> bigframes.series.Series:
        """Retrieve the md5 hash of the Blob.

        .. note::
            BigFrames Blob is still under experiments. It may not work and subject to change in the future.

        Returns:
            bigframes.series.Series: string of the md5 hash."""
        return (
            self.metadata()
            ._apply_unary_op(ops.JSONValue(json_path="$.md5_hash"))
            .rename("md5_hash")
        )

    def size(self) -> bigframes.series.Series:
        """Retrieve the file size of the Blob.

        .. note::
            BigFrames Blob is still under experiments. It may not work and subject to change in the future.

        Returns:
            bigframes.series.Series: file size in bytes."""
        return (
            self.metadata()
            ._apply_unary_op(ops.JSONValue(json_path="$.size"))
            .rename("size")
            .astype("Int64")
        )

    def updated(self) -> bigframes.series.Series:
        """Retrieve the updated time of the Blob.

        .. note::
            BigFrames Blob is still under experiments. It may not work and subject to change in the future.

        Returns:
            bigframes.series.Series: updated time as UTC datetime."""
        import bigframes.pandas as bpd

        updated = (
            self.metadata()
            ._apply_unary_op(ops.JSONValue(json_path="$.updated"))
            .rename("updated")
            .astype("Int64")
        )

        return bpd.to_datetime(updated, unit="us", utc=True)

    def _get_runtime(
        self, mode: str, with_metadata: bool = False
    ) -> bigframes.series.Series:
        """Retrieve the ObjectRefRuntime as JSON.

        Args:
            mode (str): mode for the URLs, "R" for read, "RW" for read & write.
            metadata (bool, default False): whether to fetch the metadata in the ObjectRefRuntime.

        Returns:
            bigframes.series.Series: ObjectRefRuntime JSON.
        """
        s = self._apply_unary_op(ops.obj_fetch_metadata_op) if with_metadata else self

        return s._apply_unary_op(ops.ObjGetAccessUrl(mode=mode))

    def _df_apply_udf(
        self, df: bigframes.dataframe.DataFrame, udf
    ) -> bigframes.series.Series:
        # Catch and rethrow function axis=1 warning to be more user-friendly.
        with warnings.catch_warnings(record=True) as catched_warnings:
            s = df.apply(udf, axis=1)
        for w in catched_warnings:
            if isinstance(w.message, bfe.FunctionAxisOnePreviewWarning):
                warnings.warn(
                    "Blob Functions use bigframes DataFrame Managed function with axis=1 senario, which is a preview feature.",
                    category=w.category,
                    stacklevel=2,
                )
            else:
                warnings.warn_explicit(
                    message=w.message,
                    category=w.category,
                    filename=w.filename,
                    lineno=w.lineno,
                    source=w.source,
                )

        return s

    def read_url(self) -> bigframes.series.Series:
        """Retrieve the read URL of the Blob.

        .. note::
            BigFrames Blob is still under experiments. It may not work and subject to change in the future.

        Returns:
            bigframes.series.Series: Read only URLs."""
        return self._get_runtime(mode="R")._apply_unary_op(
            ops.JSONValue(json_path="$.access_urls.read_url")
        )

    def write_url(self) -> bigframes.series.Series:
        """Retrieve the write URL of the Blob.

        .. note::
            BigFrames Blob is still under experiments. It may not work and subject to change in the future.

        Returns:
            bigframes.series.Series: Writable URLs."""
        return self._get_runtime(mode="RW")._apply_unary_op(
            ops.JSONValue(json_path="$.access_urls.write_url")
        )

    def display(self, n: int = 3, *, content_type: str = ""):
        """Display the blob content in the IPython Notebook environment. Only works for image type now.

        .. note::
            BigFrames Blob is still under experiments. It may not work and subject to change in the future.

        Args:
            n (int, default 3): number of sample blob objects to display.
            content_type (str, default ""): content type of the blob. If unset, use the blob metadata of the storage. Possible values are "image", "audio" and "video".
        """
        # col name doesn't matter here. Rename to avoid column name conflicts
        df = bigframes.series.Series(self._block).rename("blob_col").to_frame()

        df["read_url"] = df["blob_col"].blob.read_url()

        if content_type:
            df["content_type"] = content_type
        else:
            df["content_type"] = df["blob_col"].blob.content_type()

        pandas_df, _, query_job = df._block.retrieve_repr_request_results(n)
        df._set_internal_query_job(query_job)

        def display_single_url(
            read_url: str, content_type: Union[str, pd._libs.missing.NAType]
        ):
            if content_type is pd.NA:  # display as raw data or error
                response = requests.get(read_url)
                ipy_display.display(response.content)
                return

            content_type = cast(str, content_type).casefold()

            if content_type.startswith("image"):
                ipy_display.display(ipy_display.Image(url=read_url))
            elif content_type.startswith("audio"):
                # using url somehow doesn't work with audios
                response = requests.get(read_url)
                ipy_display.display(ipy_display.Audio(response.content))
            elif content_type.startswith("video"):
                ipy_display.display(ipy_display.Video(read_url))
            else:  # display as raw data
                response = requests.get(read_url)
                ipy_display.display(response.content)

        for _, row in pandas_df.iterrows():
            display_single_url(row["read_url"], row["content_type"])

    @property
    def session(self):
        return self._block.session

    def _resolve_connection(self, connection: Optional[str] = None) -> str:
        """Resovle the BigQuery connection.

        .. note::
            BigFrames Blob is still under experiments. It may not work and
            subject to change in the future.

        Args:
            connection (str or None, default None): BQ connection used for
                function internet transactions, and the output blob if "dst" is
                str. If None, uses default connection of the session.

        Returns:
            str: the resolved BigQuery connection string in the format:
             "project.location.connection_id".

        Raises:
            ValueError: If the connection cannot be resolved to a valid string.
        """
        connection = connection or self._block.session._bq_connection
        return clients.resolve_full_bq_connection_name(
            connection,
            default_project=self._block.session._project,
            default_location=self._block.session._location,
        )

    def _get_runtime_json_str(
        self, mode: str = "R", with_metadata: bool = False
    ) -> bigframes.series.Series:
        """Get the runtime and apply the ToJSONSTring transformation.

        .. note::
            BigFrames Blob is still under experiments. It may not work and
            subject to change in the future.

        Args:
            mode(str or str, default "R"): the mode for accessing the runtime.
                Default to "R". Possible values are "R" (read-only) and
                "RW" (read-write)
            with_metadata (bool, default False): whether to include metadata
                in the JOSN string. Default to False.

        Returns:
            str: the runtime object in the JSON string.
        """
        runtime = self._get_runtime(mode=mode, with_metadata=with_metadata)
        return runtime._apply_unary_op(ops.ToJSONString())

    # TODO(b/404605969): remove cleanups when UDF fixes dataset deletion.
    def _add_to_cleanup_set(self, udf):
        """Add udf name to session cleanup set. Won't need this after UDF fixes dataset deletion."""
        self.session._function_session._update_temp_artifacts(
            udf.bigframes_bigquery_function, ""
        )

    def image_blur(
        self,
        ksize: tuple[int, int],
        *,
        dst: Optional[Union[str, bigframes.series.Series]] = None,
        connection: Optional[str] = None,
        max_batching_rows: int = 8192,
        container_cpu: Union[float, int] = 0.33,
        container_memory: str = "512Mi",
    ) -> bigframes.series.Series:
        """Blurs images.

        .. note::
            BigFrames Blob is still under experiments. It may not work and subject to change in the future.

        Args:
            ksize (tuple(int, int)): Kernel size.
            dst (str or bigframes.series.Series or None, default None): Output destination. Can be one of:
                str: GCS folder str. The output filenames are the same as the input files.
                blob Series: The output file paths are determined by the uris of the blob Series.
                None: Output to BQ as bytes.
                Encoding is determined by the extension of the output filenames (or input filenames if doesn't have output filenames). If filename doesn't have an extension, use ".jpeg" for encoding.
            connection (str or None, default None): BQ connection used for function internet transactions, and the output blob if "dst" is str. If None, uses default connection of the session.
            max_batching_rows (int, default 8,192): Max number of rows per batch send to cloud run to execute the function.
            container_cpu (int or float, default 0.33): number of container CPUs. Possible values are [0.33, 8]. Floats larger than 1 are cast to intergers.
            container_memory (str, default "512Mi"): container memory size. String of the format <number><unit>. Possible values are from 512Mi to 32Gi.

        Returns:
            bigframes.series.Series: blob Series if destination is GCS. Or bytes Series if destination is BQ.
        """
        import bigframes.blob._functions as blob_func

        connection = self._resolve_connection(connection)
        df = self._get_runtime_json_str(mode="R").to_frame()

        if dst is None:
            ext = self.uri().str.extract(FILE_EXT_REGEX)

            image_blur_udf = blob_func.TransformFunction(
                blob_func.image_blur_to_bytes_def,
                session=self._block.session,
                connection=connection,
                max_batching_rows=max_batching_rows,
                container_cpu=container_cpu,
                container_memory=container_memory,
            ).udf()

            df["ksize_x"], df["ksize_y"] = ksize
            df["ext"] = ext  # type: ignore
            res = self._df_apply_udf(df, image_blur_udf)

            return res

        if isinstance(dst, str):
            dst = os.path.join(dst, "")
            # Replace src folder with dst folder, keep the file names.
            dst_uri = self.uri().str.replace(FILE_FOLDER_REGEX, rf"{dst}\1", regex=True)
            dst = cast(
                bigframes.series.Series, dst_uri.str.to_blob(connection=connection)
            )

        ext = dst.blob.uri().str.extract(FILE_EXT_REGEX)

        image_blur_udf = blob_func.TransformFunction(
            blob_func.image_blur_def,
            session=self._block.session,
            connection=connection,
            max_batching_rows=max_batching_rows,
            container_cpu=container_cpu,
            container_memory=container_memory,
        ).udf()

        dst_rt = dst.blob._get_runtime_json_str(mode="RW")

        df = df.join(dst_rt, how="outer")
        df["ksize_x"], df["ksize_y"] = ksize
        df["ext"] = ext  # type: ignore

        res = self._df_apply_udf(df, image_blur_udf)
        res.cache()  # to execute the udf

        self._add_to_cleanup_set(image_blur_udf)

        return dst

    def image_resize(
        self,
        dsize: tuple[int, int] = (0, 0),
        *,
        fx: float = 0.0,
        fy: float = 0.0,
        dst: Optional[Union[str, bigframes.series.Series]] = None,
        connection: Optional[str] = None,
        max_batching_rows: int = 8192,
        container_cpu: Union[float, int] = 0.33,
        container_memory: str = "512Mi",
    ):
        """Resize images.

        .. note::
            BigFrames Blob is still under experiments. It may not work and subject to change in the future.

        Args:
            dsize (tuple(int, int), default (0, 0)): Destination size. If set to 0, fx and fy parameters determine the size.
            fx (float, default 0.0): scale factor along the horizontal axis. If set to 0.0, dsize parameter determines the output size.
            fy (float, defalut 0.0): scale factor along the vertical axis. If set to 0.0, dsize parameter determines the output size.
            dst (str or bigframes.series.Series or None, default None): Output destination. Can be one of:
                str: GCS folder str. The output filenames are the same as the input files.
                blob Series: The output file paths are determined by the uris of the blob Series.
                None: Output to BQ as bytes.
                Encoding is determined by the extension of the output filenames (or input filenames if doesn't have output filenames). If filename doesn't have an extension, use ".jpeg" for encoding.
            connection (str or None, default None): BQ connection used for function internet transactions, and the output blob if "dst" is str. If None, uses default connection of the session.
            max_batching_rows (int, default 8,192): Max number of rows per batch send to cloud run to execute the function.
            container_cpu (int or float, default 0.33): number of container CPUs. Possible values are [0.33, 8]. Floats larger than 1 are cast to intergers.
            container_memory (str, default "512Mi"): container memory size. String of the format <number><unit>. Possible values are from 512Mi to 32Gi.

        Returns:
            bigframes.series.Series: blob Series if destination is GCS. Or bytes Series if destination is BQ.
        """
        dsize_set = dsize[0] > 0 and dsize[1] > 0
        fsize_set = fx > 0.0 and fy > 0.0
        if not dsize_set ^ fsize_set:
            raise ValueError(
                "Only one of dsize or (fx, fy) parameters must be set. And the set values must be positive. "
            )

        import bigframes.blob._functions as blob_func

        connection = self._resolve_connection(connection)
        df = self._get_runtime_json_str(mode="R").to_frame()

        if dst is None:
            ext = self.uri().str.extract(FILE_EXT_REGEX)

            image_resize_udf = blob_func.TransformFunction(
                blob_func.image_resize_to_bytes_def,
                session=self._block.session,
                connection=connection,
                max_batching_rows=max_batching_rows,
                container_cpu=container_cpu,
                container_memory=container_memory,
            ).udf()

            df["dsize_x"], df["dsizye_y"] = dsize
            df["fx"], df["fy"] = fx, fy
            df["ext"] = ext  # type: ignore
            res = self._df_apply_udf(df, image_resize_udf)

            return res

        if isinstance(dst, str):
            dst = os.path.join(dst, "")
            # Replace src folder with dst folder, keep the file names.
            dst_uri = self.uri().str.replace(FILE_FOLDER_REGEX, rf"{dst}\1", regex=True)
            dst = cast(
                bigframes.series.Series, dst_uri.str.to_blob(connection=connection)
            )

        ext = dst.blob.uri().str.extract(FILE_EXT_REGEX)

        image_resize_udf = blob_func.TransformFunction(
            blob_func.image_resize_def,
            session=self._block.session,
            connection=connection,
            max_batching_rows=max_batching_rows,
            container_cpu=container_cpu,
            container_memory=container_memory,
        ).udf()

        dst_rt = dst.blob._get_runtime_json_str(mode="RW")

        df = df.join(dst_rt, how="outer")
        df["dsize_x"], df["dsizye_y"] = dsize
        df["fx"], df["fy"] = fx, fy
        df["ext"] = ext  # type: ignore

        res = self._df_apply_udf(df, image_resize_udf)
        res.cache()  # to execute the udf

        self._add_to_cleanup_set(image_resize_udf)

        return dst

    def image_normalize(
        self,
        *,
        alpha: float = 1.0,
        beta: float = 0.0,
        norm_type: str = "l2",
        dst: Optional[Union[str, bigframes.series.Series]] = None,
        connection: Optional[str] = None,
        max_batching_rows: int = 8192,
        container_cpu: Union[float, int] = 0.33,
        container_memory: str = "512Mi",
    ) -> bigframes.series.Series:
        """Normalize images.

        .. note::
            BigFrames Blob is still under experiments. It may not work and subject to change in the future.

        Args:
            alpha (float, default 1.0): Norm value to normalize to or the lower range boundary in case of the range normalization.
            beta (float, default 0.0): Upper range boundary in case of the range normalization; it is not used for the norm normalization.
            norm_type (str, default "l2"): Normalization type. Accepted values are "inf", "l1", "l2" and "minmax".
            dst (str or bigframes.series.Series or None, default None): Output destination. Can be one of:
                str: GCS folder str. The output filenames are the same as the input files.
                blob Series: The output file paths are determined by the uris of the blob Series.
                None: Output to BQ as bytes.
                Encoding is determined by the extension of the output filenames (or input filenames if doesn't have output filenames). If filename doesn't have an extension, use ".jpeg" for encoding.
            connection (str or None, default None): BQ connection used for function internet transactions, and the output blob if "dst" is str. If None, uses default connection of the session.
            max_batching_rows (int, default 8,192): Max number of rows per batch send to cloud run to execute the function.
            container_cpu (int or float, default 0.33): number of container CPUs. Possible values are [0.33, 8]. Floats larger than 1 are cast to intergers.
            container_memory (str, default "512Mi"): container memory size. String of the format <number><unit>. Possible values are from 512Mi to 32Gi.

        Returns:
            bigframes.series.Series: blob Series if destination is GCS. Or bytes Series if destination is BQ.
        """
        import bigframes.blob._functions as blob_func

        connection = self._resolve_connection(connection)
        df = self._get_runtime_json_str(mode="R").to_frame()

        if dst is None:
            ext = self.uri().str.extract(FILE_EXT_REGEX)

            image_normalize_udf = blob_func.TransformFunction(
                blob_func.image_normalize_to_bytes_def,
                session=self._block.session,
                connection=connection,
                max_batching_rows=max_batching_rows,
                container_cpu=container_cpu,
                container_memory=container_memory,
            ).udf()

            df["alpha"] = alpha
            df["beta"] = beta
            df["norm_type"] = norm_type
            df["ext"] = ext  # type: ignore
            res = self._df_apply_udf(df, image_normalize_udf)

            return res

        if isinstance(dst, str):
            dst = os.path.join(dst, "")
            # Replace src folder with dst folder, keep the file names.
            dst_uri = self.uri().str.replace(FILE_FOLDER_REGEX, rf"{dst}\1", regex=True)
            dst = cast(
                bigframes.series.Series, dst_uri.str.to_blob(connection=connection)
            )

        ext = dst.blob.uri().str.extract(FILE_EXT_REGEX)

        image_normalize_udf = blob_func.TransformFunction(
            blob_func.image_normalize_def,
            session=self._block.session,
            connection=connection,
            max_batching_rows=max_batching_rows,
            container_cpu=container_cpu,
            container_memory=container_memory,
        ).udf()

        dst_rt = dst.blob._get_runtime_json_str(mode="RW")

        df = df.join(dst_rt, how="outer")
        df["alpha"] = alpha
        df["beta"] = beta
        df["norm_type"] = norm_type
        df["ext"] = ext  # type: ignore

        res = self._df_apply_udf(df, image_normalize_udf)
        res.cache()  # to execute the udf

        self._add_to_cleanup_set(image_normalize_udf)

        return dst

    def pdf_extract(
        self,
        *,
        connection: Optional[str] = None,
        max_batching_rows: int = 1,
        container_cpu: Union[float, int] = 2,
        container_memory: str = "1Gi",
        verbose: bool = False,
    ) -> bigframes.series.Series:
        """Extracts text from PDF URLs and saves the text as string.

        .. note::
            BigFrames Blob is still under experiments. It may not work and
            subject to change in the future.

        Args:
            connection (str or None, default None): BQ connection used for
                function internet transactions, and the output blob if "dst"
                is str. If None, uses default connection of the session.
            max_batching_rows (int, default 1): Max number of rows per batch
                send to cloud run to execute the function.
            container_cpu (int or float, default 2): number of container CPUs. Possible values are [0.33, 8]. Floats larger than 1 are cast to intergers.
            container_memory (str, default "1Gi"): container memory size. String of the format <number><unit>. Possible values are from 512Mi to 32Gi.
            verbose (bool, default "False"): controls the verbosity of the output.
                When set to True, both error messages and the extracted content
                are displayed. Conversely, when set to False, only the extracted
                content is presented, suppressing error messages.

        Returns:
            bigframes.series.Series: str or struct[str, str],
                depend on the "verbose" parameter.
                Contains the extracted text from the PDF file.
                Includes error messages if verbosity is enabled.
        """
        import bigframes.bigquery as bbq
        import bigframes.blob._functions as blob_func
        import bigframes.pandas as bpd

        connection = self._resolve_connection(connection)

        pdf_extract_udf = blob_func.TransformFunction(
            blob_func.pdf_extract_def,
            session=self._block.session,
            connection=connection,
            max_batching_rows=max_batching_rows,
            container_cpu=container_cpu,
            container_memory=container_memory,
        ).udf()

        src_rt = self._get_runtime_json_str(mode="R")

        res = src_rt.apply(pdf_extract_udf)

        content_series = res._apply_unary_op(ops.JSONValue(json_path="$.content"))

        self._add_to_cleanup_set(pdf_extract_udf)
        if verbose:
            status_series = res._apply_unary_op(ops.JSONValue(json_path="$.status"))
            res_df = bpd.DataFrame({"status": status_series, "content": content_series})
            struct_series = bbq.struct(res_df)
            return struct_series
        else:
            return content_series

    def pdf_chunk(
        self,
        *,
        connection: Optional[str] = None,
        chunk_size: int = 2000,
        overlap_size: int = 200,
        max_batching_rows: int = 1,
        container_cpu: Union[float, int] = 2,
        container_memory: str = "1Gi",
        verbose: bool = False,
    ) -> bigframes.series.Series:
        """Extracts and chunks text from PDF URLs and saves the text as
           arrays of strings.

        .. note::
            BigFrames Blob is still under experiments. It may not work and subject to change in the future.

        Args:
            connection (str or None, default None): BQ connection used for
                function internet transactions, and the output blob if "dst"
                is str. If None, uses default connection of the session.
            chunk_size (int, default 2000): the desired size of each text chunk
                (number of characters).
            overlap_size (int, default 200): the number of overlapping characters
                between consective chunks. The helps to ensure context is
                perserved across chunk boundaries.
            max_batching_rows (int, default 1): Max number of rows per batch
                send to cloud run to execute the function.
            container_cpu (int or float, default 2): number of container CPUs. Possible values are [0.33, 8]. Floats larger than 1 are cast to intergers.
            container_memory (str, default "1Gi"): container memory size. String of the format <number><unit>. Possible values are from 512Mi to 32Gi.
            verbose (bool, default "False"): controls the verbosity of the output.
                When set to True, both error messages and the extracted content
                are displayed. Conversely, when set to False, only the extracted
                content is presented, suppressing error messages.

        Returns:
            bigframe.series.Series: array[str] or struct[str, array[str]],
                depend on the "verbose" parameter.
                where each string is a chunk of text extracted from PDF.
                Includes error messages if verbosity is enabled.
        """

        import bigframes.bigquery as bbq
        import bigframes.blob._functions as blob_func
        import bigframes.pandas as bpd

        connection = self._resolve_connection(connection)

        if chunk_size <= 0:
            raise ValueError("chunk_size must be a positive integer.")
        if overlap_size < 0:
            raise ValueError("overlap_size must be a non-negative integer.")
        if overlap_size >= chunk_size:
            raise ValueError("overlap_size must be smaller than chunk_size.")

        pdf_chunk_udf = blob_func.TransformFunction(
            blob_func.pdf_chunk_def,
            session=self._block.session,
            connection=connection,
            max_batching_rows=max_batching_rows,
            container_cpu=container_cpu,
            container_memory=container_memory,
        ).udf()

        src_rt = self._get_runtime_json_str(mode="R")
        df = src_rt.to_frame()
        df["chunk_size"] = chunk_size
        df["overlap_size"] = overlap_size

        res = self._df_apply_udf(df, pdf_chunk_udf)

        content_series = bbq.json_extract_string_array(res, "$.content")
        self._add_to_cleanup_set(pdf_chunk_udf)
        if verbose:
            status_series = res._apply_unary_op(ops.JSONValue(json_path="$.status"))
            res_df = bpd.DataFrame({"status": status_series, "content": content_series})
            struct_series = bbq.struct(res_df)
            return struct_series
        else:
            return content_series
