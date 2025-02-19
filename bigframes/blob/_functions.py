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

from dataclasses import dataclass
import inspect
from typing import Callable, Iterable

import google.cloud.bigquery as bigquery

import bigframes.session
import bigframes.session._io.bigquery as bf_io_bigquery

_PYTHON_TO_BQ_TYPES = {int: "INT64", float: "FLOAT64", str: "STRING", bytes: "BYTES"}


@dataclass(frozen=True)
class FunctionDef:
    """Definition of a Python UDF."""

    func: Callable  # function body
    requirements: Iterable[str]  # required packages


# TODO(garrettwu): migrate to bigframes UDF when it is available
class TransformFunction:
    """Simple transform function class to deal with Python UDF."""

    def __init__(
        self,
        func_def: FunctionDef,
        session: bigframes.session.Session,
        connection: str,
        max_batching_rows: int,
    ):
        self._func = func_def.func
        self._requirements = func_def.requirements
        self._session = session
        self._connection = connection
        self._max_batching_rows = max_batching_rows

    def _input_bq_signature(self):
        sig = inspect.signature(self._func)
        inputs = []
        for k, v in sig.parameters.items():
            inputs.append(f"{k} {_PYTHON_TO_BQ_TYPES[v.annotation]}")
        return ", ".join(inputs)

    def _output_bq_type(self):
        sig = inspect.signature(self._func)
        return _PYTHON_TO_BQ_TYPES[sig.return_annotation]

    def _create_udf(self):
        """Create Python UDF in BQ. Return name of the UDF."""
        udf_name = str(self._session._loader._storage_manager._random_table())

        func_body = inspect.getsource(self._func)
        func_name = self._func.__name__
        packages = str(list(self._requirements))

        sql = f"""
CREATE OR REPLACE FUNCTION `{udf_name}`({self._input_bq_signature()})
RETURNS {self._output_bq_type()} LANGUAGE python
WITH CONNECTION `{self._connection}`
OPTIONS (entry_point='{func_name}', runtime_version='python-3.11', packages={packages}, max_batching_rows={self._max_batching_rows})
AS r\"\"\"


{func_body}


\"\"\"
        """

        bf_io_bigquery.start_query_with_client(
            self._session.bqclient,
            sql,
            job_config=bigquery.QueryJobConfig(),
            metrics=self._session._metrics,
        )

        return udf_name

    def udf(self):
        """Create and return the UDF object."""
        udf_name = self._create_udf()
        return self._session.read_gbq_function(udf_name)


# Blur images. Takes ObjectRefRuntime as JSON string. Outputs ObjectRefRuntime JSON string.
def image_blur_func(
    src_obj_ref_rt: str, dst_obj_ref_rt: str, ksize_x: int, ksize_y: int
) -> str:
    import json

    import cv2 as cv  # type: ignore
    import numpy as np
    import requests

    src_obj_ref_rt_json = json.loads(src_obj_ref_rt)
    dst_obj_ref_rt_json = json.loads(dst_obj_ref_rt)

    src_url = src_obj_ref_rt_json["access_urls"]["read_url"]
    dst_url = dst_obj_ref_rt_json["access_urls"]["write_url"]

    response = requests.get(src_url)
    bts = response.content

    nparr = np.frombuffer(bts, np.uint8)
    img = cv.imdecode(nparr, cv.IMREAD_UNCHANGED)
    img_blurred = cv.blur(img, ksize=(ksize_x, ksize_y))
    bts = cv.imencode(".jpeg", img_blurred)[1].tobytes()

    requests.put(
        url=dst_url,
        data=bts,
        headers={
            "Content-Type": "image/jpeg",
        },
    )

    return dst_obj_ref_rt


image_blur_def = FunctionDef(image_blur_func, ["opencv-python", "numpy", "requests"])


def image_blur_to_bytes_func(src_obj_ref_rt: str, ksize_x: int, ksize_y: int) -> bytes:
    import json

    import cv2 as cv  # type: ignore
    import numpy as np
    import requests

    src_obj_ref_rt_json = json.loads(src_obj_ref_rt)
    src_url = src_obj_ref_rt_json["access_urls"]["read_url"]

    response = requests.get(src_url)
    bts = response.content

    nparr = np.frombuffer(bts, np.uint8)
    img = cv.imdecode(nparr, cv.IMREAD_UNCHANGED)
    img_blurred = cv.blur(img, ksize=(ksize_x, ksize_y))
    bts = cv.imencode(".jpeg", img_blurred)[1].tobytes()

    return bts


image_blur_to_bytes_def = FunctionDef(
    image_blur_to_bytes_func, ["opencv-python", "numpy", "requests"]
)


def image_resize_func(
    src_obj_ref_rt: str,
    dst_obj_ref_rt: str,
    dsize_x: int,
    dsize_y: int,
    fx: float,
    fy: float,
) -> str:
    import json

    import cv2 as cv  # type: ignore
    import numpy as np
    import requests

    src_obj_ref_rt_json = json.loads(src_obj_ref_rt)
    dst_obj_ref_rt_json = json.loads(dst_obj_ref_rt)

    src_url = src_obj_ref_rt_json["access_urls"]["read_url"]
    dst_url = dst_obj_ref_rt_json["access_urls"]["write_url"]

    response = requests.get(src_url)
    bts = response.content

    nparr = np.frombuffer(bts, np.uint8)
    img = cv.imdecode(nparr, cv.IMREAD_UNCHANGED)
    img_resized = cv.resize(img, dsize=(dsize_x, dsize_y), fx=fx, fy=fy)
    bts = cv.imencode(".jpeg", img_resized)[1].tobytes()

    requests.put(
        url=dst_url,
        data=bts,
        headers={
            "Content-Type": "image/jpeg",
        },
    )

    return dst_obj_ref_rt


image_resize_def = FunctionDef(
    image_resize_func, ["opencv-python", "numpy", "requests"]
)


def image_resize_to_bytes_func(
    src_obj_ref_rt: str,
    dsize_x: int,
    dsize_y: int,
    fx: float,
    fy: float,
) -> bytes:
    import json

    import cv2 as cv  # type: ignore
    import numpy as np
    import requests

    src_obj_ref_rt_json = json.loads(src_obj_ref_rt)
    src_url = src_obj_ref_rt_json["access_urls"]["read_url"]

    response = requests.get(src_url)
    bts = response.content

    nparr = np.frombuffer(bts, np.uint8)
    img = cv.imdecode(nparr, cv.IMREAD_UNCHANGED)
    img_resized = cv.resize(img, dsize=(dsize_x, dsize_y), fx=fx, fy=fy)
    bts = cv.imencode(".jpeg", img_resized)[1].tobytes()

    return bts


image_resize_to_bytes_def = FunctionDef(
    image_resize_to_bytes_func, ["opencv-python", "numpy", "requests"]
)


def image_normalize_func(
    src_obj_ref_rt: str, dst_obj_ref_rt: str, alpha: float, beta: float, norm_type: str
) -> str:
    import json

    import cv2 as cv  # type: ignore
    import numpy as np
    import requests

    norm_type_mapping = {
        "inf": cv.NORM_INF,
        "l1": cv.NORM_L1,
        "l2": cv.NORM_L2,
        "minmax": cv.NORM_MINMAX,
    }

    src_obj_ref_rt_json = json.loads(src_obj_ref_rt)
    dst_obj_ref_rt_json = json.loads(dst_obj_ref_rt)

    src_url = src_obj_ref_rt_json["access_urls"]["read_url"]
    dst_url = dst_obj_ref_rt_json["access_urls"]["write_url"]

    response = requests.get(src_url)
    bts = response.content

    nparr = np.frombuffer(bts, np.uint8)
    img = cv.imdecode(nparr, cv.IMREAD_UNCHANGED)
    img_normalized = cv.normalize(
        img, None, alpha=alpha, beta=beta, norm_type=norm_type_mapping[norm_type]
    )
    bts = cv.imencode(".jpeg", img_normalized)[1].tobytes()

    requests.put(
        url=dst_url,
        data=bts,
        headers={
            "Content-Type": "image/jpeg",
        },
    )

    return dst_obj_ref_rt


image_normalize_def = FunctionDef(
    image_normalize_func, ["opencv-python", "numpy", "requests"]
)


def image_normalize_to_bytes_func(
    src_obj_ref_rt: str, alpha: float, beta: float, norm_type: str
) -> bytes:
    import json

    import cv2 as cv  # type: ignore
    import numpy as np
    import requests

    norm_type_mapping = {
        "inf": cv.NORM_INF,
        "l1": cv.NORM_L1,
        "l2": cv.NORM_L2,
        "minmax": cv.NORM_MINMAX,
    }

    src_obj_ref_rt_json = json.loads(src_obj_ref_rt)
    src_url = src_obj_ref_rt_json["access_urls"]["read_url"]

    response = requests.get(src_url)
    bts = response.content

    nparr = np.frombuffer(bts, np.uint8)
    img = cv.imdecode(nparr, cv.IMREAD_UNCHANGED)
    img_normalized = cv.normalize(
        img, None, alpha=alpha, beta=beta, norm_type=norm_type_mapping[norm_type]
    )
    bts = cv.imencode(".jpeg", img_normalized)[1].tobytes()

    return bts


image_normalize_to_bytes_def = FunctionDef(
    image_normalize_to_bytes_func, ["opencv-python", "numpy", "requests"]
)


# Extracts all text from a PDF url
def pdf_extract_func(src_obj_ref_rt: str) -> str:
    import io
    import json

    from pypdf import PdfReader  # type: ignore
    import requests

    src_obj_ref_rt_json = json.loads(src_obj_ref_rt)
    src_url = src_obj_ref_rt_json["access_urls"]["read_url"]

    response = requests.get(src_url, stream=True)
    response.raise_for_status()
    pdf_bytes = response.content

    pdf_file = io.BytesIO(pdf_bytes)
    reader = PdfReader(pdf_file, strict=False)

    all_text = ""
    for page in reader.pages:
        page_extract_text = page.extract_text()
        if page_extract_text:
            all_text += page_extract_text
    return all_text


pdf_extract_def = FunctionDef(pdf_extract_func, ["pypdf", "requests"])


# Extracts text from a PDF url and chunks it simultaneously
def pdf_chunk_func(src_obj_ref_rt: str, chunk_size: int, overlap_size: int) -> str:
    import io
    import json

    from pypdf import PdfReader  # type: ignore
    import requests

    src_obj_ref_rt_json = json.loads(src_obj_ref_rt)
    src_url = src_obj_ref_rt_json["access_urls"]["read_url"]

    response = requests.get(src_url, stream=True)
    response.raise_for_status()
    pdf_bytes = response.content

    pdf_file = io.BytesIO(pdf_bytes)
    reader = PdfReader(pdf_file, strict=False)

    # extract and chunk text simultaneously
    all_text_chunks = []
    curr_chunk = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            curr_chunk += page_text
            # split the accumulated text into chunks of a specific size with overlaop
            # this loop implements a sliding window approach to create chunks
            while len(curr_chunk) >= chunk_size:
                split_idx = curr_chunk.rfind(" ", 0, chunk_size)
                if split_idx == -1:
                    split_idx = chunk_size
                actual_chunk = curr_chunk[:split_idx]
                all_text_chunks.append(actual_chunk)
                overlap = curr_chunk[split_idx + 1 : split_idx + 1 + overlap_size]
                curr_chunk = overlap + curr_chunk[split_idx + 1 + overlap_size :]
    if curr_chunk:
        all_text_chunks.append(curr_chunk)

    all_text_json_string = json.dumps(all_text_chunks)
    return all_text_json_string


pdf_chunk_def = FunctionDef(pdf_chunk_func, ["pypdf", "requests"])
