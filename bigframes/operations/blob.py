from IPython.display import display, Image
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
        return df.merge(master_df, how="left", on="uri")

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
            bs = self._gcs_manager.download_as_bytes(bucket, path)
            display(Image(bs))
