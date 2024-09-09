from IPython.display import display, Image
import parse

from bigframes import clients
from bigframes.operations import base


class BlobMethods(base.SeriesMethods):
    # def __init__(self):
    #     self._gcs_manager = clients.GcsManager()

    def _get_merged_df(self):
        import bigframes.pandas as bpd

        session = self._block.session
        master_object_table = session._master_object_table

        master_df = session.read_gbq(master_object_table)
        df = bpd.DataFrame(self._block)
        return df.merge(master_df, how="left", left_on=df.columns[0], right_on="uri")

    def version(self):
        merged_df = self._get_merged_df()

        return merged_df["generation"].rename("version")

    def content_type(self):
        merged_df = self._get_merged_df()

        return merged_df["content_type"]

    def md5_hash(self):
        merged_df = self._get_merged_df()

        return merged_df["md5_hash"]

    def display(self, n=3):
        import bigframes.pandas as bpd

        self._gcs_manager = clients.GcsManager()
        s = bpd.Series(self._block)
        s = s.head(n)

        for uri in s:
            (bucket, path) = parse_gcs_path(uri)
            bts = self._gcs_manager.download_as_bytes(bucket, path)
            display(Image(bts))

    def copy(self, dst_folder, times=1):
        import bigframes.pandas as bpd

        s = bpd.Series(self._block)
        session = self._block.session

        @session.remote_function(
            [str],
            str,
            packages=["google-cloud-storage", "parse"],
            max_batching_rows=50,
        )
        def copy_files(uri_in):
            import os

            from google.cloud import storage
            import parse

            storage_client = storage.Client()

            bucket_name, blob_path_in = parse.parse("gs://{0}/{1}", uri_in)
            bucket = storage_client.bucket(bucket_name)
            blob_in = bucket.blob(blob_path_in)
            bts = blob_in.download_as_bytes()

            file_name_full = uri_in[uri_in.rfind("/") + 1 :]
            file_name, ext = os.path.splitext(file_name_full)
            for i in range(times):
                out_file_name = file_name + str(i) + ext
                blob_path_out = os.path.join(dst_folder, out_file_name)
                blob_out = bucket.blob(blob_path_out)
                blob_out.upload_from_string(bts)

            return str(times)

        return s.apply(copy_files)

    def img_rm_background(self, dst_folder, random_file=False):
        import bigframes.pandas as bpd

        s = bpd.Series(self._block)
        session = self._block.session
        session._transform_output_obj_table = session._create_object_table(
            "gs://garrettwu_bucket/" + dst_folder + "/*"
        )

        @session.remote_function(
            [str],
            str,
            packages=["google-cloud-storage", "rembg", "parse"],
            max_batching_rows=50,
            cloud_function_memory_mib=32768,
        )
        def bigframes_img_rm_bg(uri_in):
            import os

            from google.cloud import storage
            import parse
            from rembg import remove

            def rand_file_name(file_name_full):
                import uuid

                file_name, ext = os.path.splitext(file_name_full)
                return file_name + uuid.uuid4().hex + ext

            storage_client = storage.Client()

            bucket_name, blob_path_in = parse.parse("gs://{0}/{1}", uri_in)
            bucket = storage_client.bucket(bucket_name)
            blob_in = bucket.blob(blob_path_in)
            bts = blob_in.download_as_bytes()

            bts = remove(bts)

            file_name_full = uri_in[uri_in.rfind("/") + 1 :]
            if random_file:
                file_name_full = rand_file_name(file_name_full)

            blob_path_out = os.path.join(dst_folder, file_name_full)
            blob_out = bucket.blob(blob_path_out)
            blob_out.upload_from_string(bts, content_type="image/png")

            return f"gs://{bucket_name}/{blob_path_out}"

        return s.apply(bigframes_img_rm_bg)

    # def img_rm_background1(self, dst_folder, random_file=False):
    #     import bigframes.pandas as bpd
    #     s = bpd.Series(self._block)
    #     session = self._block.session
    #     # session = bpd.get_global_session()

    #     @session.remote_function(
    #             [str],
    #             str,
    #             packages=["google-cloud-storage", "rembg", "parse"],
    #             max_batching_rows=50,
    #             cloud_function_memory_mib=32768
    #         )
    #     def bigframes_img_rm_bg1(uri_in):
    #         from rembg import remove
    #         from google.cloud import storage
    #         import parse
    #         import os

    #         def rand_file_name(file_name_full):
    #             import uuid
    #             file_name, ext = os.path.splitext(file_name_full)
    #             return file_name + uuid.uuid4().hex + ext

    #         storage_client = storage.Client()

    #         bucket_name, blob_path_in = parse.parse("gs://{0}/{1}", uri_in)
    #         bucket = storage_client.bucket(bucket_name)
    #         blob_in = bucket.blob(blob_path_in)
    #         bts = blob_in.download_as_bytes()

    #         bts = remove(bts)

    #         file_name_full = uri_in[uri_in.rfind("/") + 1 :]
    #         if random_file:
    #             file_name_full = rand_file_name(file_name_full)

    #         blob_path_out = os.path.join(dst_folder, file_name_full)
    #         blob_out = bucket.blob(blob_path_out)
    #         blob_out.upload_from_string(bts)

    #         # return f"gs://{bucket_name}/{blob_path_out}"
    #         return uri_in

    #     return s.apply(bigframes_img_rm_bg1)

    # def _bytes_to_cv_img(self, bts):
    #     nparr = np.frombuffer(bts, np.uint8)
    #     return cv.imdecode(nparr, cv.IMREAD_UNCHANGED)

    # def _cv_img_to_jpeg_bytes(self, img):
    #     return cv.imencode(".jpeg", img)[1].tobytes()

    # def _img_blur_local(self, uri, ksize: tuple[int, int], dst_folder):
    #     (bucket, path) = parse_gcs_path(uri)
    #     bts = self._gcs_manager.download_as_bytes(bucket, path)
    #     img = self._bytes_to_cv_img(bts)
    #     img_blurred = cv.blur(img, ksize)

    #     bts = self._cv_img_to_jpeg_bytes(img_blurred)

    #     file_name = uri[uri.rfind("/") + 1 :]
    #     dst_path = dst_folder + "/" + file_name

    #     return self._gcs_manager.upload_bytes(
    #         bts, bucket, dst_path, content_type="image/jpeg"
    #     )

    # def _img_blur_remote(self, s, ksize: tuple[int, int], dst_folder):
    #     session = self._block.session

    #     @session.remote_function(
    #         [str],
    #         str,
    #         packages=["numpy", "google-cloud-storage", "parse", "opencv-python"],
    #         max_batching_rows=50,
    #     )
    #     def bigframes_img_blur(uri_in):
    #         import os

    #         import cv2 as cv
    #         from google.cloud import storage
    #         import numpy as np
    #         import parse

    #         storage_client = storage.Client()

    #         bucket_name, blob_path_in = parse.parse("gs://{0}/{1}", uri_in)
    #         bucket = storage_client.bucket(bucket_name)
    #         blob_in = bucket.blob(blob_path_in)
    #         bts = blob_in.download_as_bytes()

    #         nparr = np.frombuffer(bts, np.uint8)
    #         img = cv.imdecode(nparr, cv.IMREAD_UNCHANGED)
    #         img_blurred = cv.blur(img, ksize)
    #         bts = cv.imencode(".jpeg", img_blurred)[1].tobytes()

    #         file_name_full = uri_in[uri_in.rfind("/") + 1 :]

    #         # import uuid
    #         # file_name, ext = os.path.splitext(file_name_full)
    #         # file_name_full = file_name + uuid.uuid4().hex + ext

    #         blob_path_out = os.path.join(dst_folder, file_name_full)
    #         blob_out = bucket.blob(blob_path_out)
    #         blob_out.upload_from_string(bts)

    #         return f"gs://{bucket_name}/{blob_path_out}"

    #     return s.apply(bigframes_img_blur)

    # def img_blur(self, ksize, dst_folder, mode="local"):
    #     import bigframes.pandas as bpd

    #     s = bpd.Series(self._block)
    #     if mode == "local":
    #         self._gcs_manager = clients.GcsManager()
    #         new_uris = []
    #         for uri in s:
    #             new_uri = self._img_blur_local(uri, ksize, dst_folder)
    #             new_uris.append(new_uri)

    #         return bpd.Series(new_uris)
    #     elif mode == "remote":
    #         return self._img_blur_remote(s, ksize, dst_folder)
    #     else:
    #         raise ValueError("Unsupported mode.")

    # def img_remove_background(self, dst_folder):
    #     import bigframes.pandas as bpd
    #     s = bpd.Series(self._block)
    #     session = self._block.session

    #     @session.remote_function(
    #         [str],
    #         str,
    #         packages=["numpy", "google-cloud-storage", "backgroundremover"],
    #         max_batching_rows=50,
    #     )
    #     def bigframes_img_blur(uri_in):
    #         import os

    #         from google.cloud import storage
    #         from backgroundremover.bg import remove

    #         storage_client = storage.Client()

    #         bucket_name, blob_path_in = parse.parse("gs://{0}/{1}", uri_in)
    #         bucket = storage_client.bucket(bucket_name)
    #         blob_in = bucket.blob(blob_path_in)
    #         bts = blob_in.download_as_bytes()

    #         model_choices = ["u2net", "u2net_human_seg", "u2netp"]

    #         bts = remove(bts, model_name=model_choices[0],
    #              alpha_matting=True,
    #              alpha_matting_foreground_threshold=240,
    #              alpha_matting_background_threshold=10,
    #              alpha_matting_erode_structure_size=10,
    #              alpha_matting_base_size=1000)

    #         file_name_full = uri_in[uri_in.rfind("/") + 1 :]

    #         # import uuid
    #         # file_name, ext = os.path.splitext(file_name_full)
    #         # file_name_full = file_name + uuid.uuid4().hex + ext

    #         blob_path_out = os.path.join(dst_folder, file_name_full)
    #         blob_out = bucket.blob(blob_path_out)
    #         blob_out.upload_from_string(bts)

    #         return f"gs://{bucket_name}/{blob_path_out}"

    #     return s.apply(bigframes_img_blur)

    # def _text_chunk_local(self, uri_in):
    #     import json

    #     from google.cloud import storage
    #     from llama_index.core import SimpleDirectoryReader
    #     from llama_index.core.node_parser import SentenceSplitter
    #     import parse

    #     storage_client = storage.Client()
    #     bucket_name, blob_path_in = parse.parse("gs://{0}/{1}", uri_in)

    #     bucket = storage_client.bucket(bucket_name)
    #     blob_in = bucket.blob(blob_path_in)

    #     file_name_full = uri_in[uri_in.rfind("/") + 1 :]

    #     tmp_file_path = f"/tmp/{file_name_full}"

    #     blob_in.download_to_filename(tmp_file_path)

    #     documents = SimpleDirectoryReader(input_files=[tmp_file_path]).load_data()

    #     base_splitter = SentenceSplitter(chunk_size=512)

    #     nodes = base_splitter.get_nodes_from_documents(documents)

    #     texts = [node.text for node in nodes]

    #     return json.dumps(texts)

    # def _text_chunk_remote(self):
    #     import bigframes.ml.json as bf_json
    #     import bigframes.pandas as bpd

    #     session = self._block.session

    #     @session.remote_function(
    #         [str],
    #         str,
    #         packages=[
    #             "google-cloud-storage",
    #             "parse",
    #             "llama-index-core",
    #             "llama-index-readers-file",
    #         ],
    #         max_batching_rows=10,
    #     )
    #     def bigframes_llama_index_chunk(uri_in):
    #         import json

    #         from google.cloud import storage
    #         from llama_index.core import SimpleDirectoryReader
    #         from llama_index.core.node_parser import SentenceSplitter
    #         import parse

    #         storage_client = storage.Client()
    #         bucket_name, blob_path_in = parse.parse("gs://{0}/{1}", uri_in)

    #         bucket = storage_client.bucket(bucket_name)
    #         blob_in = bucket.blob(blob_path_in)

    #         file_name_full = uri_in[uri_in.rfind("/") + 1 :]

    #         tmp_file_path = f"/tmp/{file_name_full}"

    #         blob_in.download_to_filename(tmp_file_path)

    #         documents = SimpleDirectoryReader(input_files=[tmp_file_path]).load_data()

    #         base_splitter = SentenceSplitter(chunk_size=512)

    #         nodes = base_splitter.get_nodes_from_documents(documents)

    #         texts = [node.text for node in nodes]
    #         return json.dumps(texts)

    #     s = bpd.Series(self._block)
    #     json_str_s = s.apply(bigframes_llama_index_chunk)
    #     text_arr_s = bf_json.json_extract_array(json_str_s)

    #     return text_arr_s

    # def llama_index_chunk(self, mode="local"):
    #     import bigframes.ml.json as bf_json
    #     import bigframes.pandas as bpd

    #     s = bpd.Series(self._block)
    #     if mode == "local":
    #         self._gcs_manager = clients.GcsManager()
    #         new_json_chunks = []
    #         for uri in s:
    #             json_chunks = self._text_chunk_local(uri_in=uri)
    #             new_json_chunks.append(json_chunks)

    #         json_str_df = bpd.DataFrame({"json": new_json_chunks})
    #         text_arr_s = bf_json.json_extract_array(json_str_df)

    #         return text_arr_s
    #     elif mode == "remote":
    #         return self._text_chunk_remote()
    #     else:
    #         raise ValueError("Unsupported mode.")


def parse_gcs_path(path):
    result = parse.parse("gs://{0}/{1}", path)

    return tuple(result)
