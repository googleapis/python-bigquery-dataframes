from bigframes import dataframe
from bigframes.operations import base


class BlobMethods(base.SeriesMethods):
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
