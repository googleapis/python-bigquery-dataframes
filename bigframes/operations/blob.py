import cv2 as cv
from IPython.display import display, Image
import numpy as np
import parse

from bigframes import clients, dataframe, series
from bigframes.operations import base


class BlobMethods(base.SeriesMethods):
    # def __init__(self):
    #     self._gcs_manager = clients.GcsManager()

    def _get_merged_df(self):
        session = self._block.session
        master_object_table = session._master_object_table

        master_df = session.read_gbq(master_object_table)
        df = dataframe.DataFrame(self._block)
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

    def _parse_gcs_path(self, path):
        result = parse.parse("gs://{0}/{1}", path)

        return tuple(result)

    def display(self):
        self._gcs_manager = clients.GcsManager()
        s = series.Series(self._block)
        for uri in s:
            (bucket, path) = self._parse_gcs_path(uri)
            bts = self._gcs_manager.download_as_bytes(bucket, path)
            display(Image(bts))

    def _bytes_to_cv_img(self, bts):
        nparr = np.frombuffer(bts, np.uint8)
        return cv.imdecode(nparr, cv.IMREAD_UNCHANGED)

    def _cv_img_to_jpeg_bytes(self, img):
        return cv.imencode(".jpeg", img)[1].tobytes()

    def _img_blur(self, uri, ksize: tuple[int, int], dst_folder):
        (bucket, path) = self._parse_gcs_path(uri)
        bts = self._gcs_manager.download_as_bytes(bucket, path)
        img = self._bytes_to_cv_img(bts)
        img_blurred = cv.blur(img, ksize)

        bts = self._cv_img_to_jpeg_bytes(img_blurred)

        file_name = uri[uri.rfind("/") + 1 :]
        dst_path = dst_folder + "/" + file_name

        return self._gcs_manager.upload_bytes(
            bts, bucket, dst_path, content_type="image/jpeg"
        )

    def img_blur(self, ksize, dst_folder):
        self._gcs_manager = clients.GcsManager()
        s = series.Series(self._block)
        new_uris = []
        for uri in s:
            new_uri = self._img_blur(uri, ksize, dst_folder)
            new_uris.append(new_uri)

        return series.Series(new_uris)
