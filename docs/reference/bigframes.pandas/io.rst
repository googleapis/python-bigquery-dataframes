
============
Input/output
============

.. currentmodule:: bigframes.pandas

Pandas
------

.. autofunction:: read_pandas

See also:

* :meth:`bigframes.session.Session.read_pandas`
* :meth:`bigframes.dataframe.DataFrame.to_pandas`

Flat file
---------

.. autofunction:: read_csv

See also:

* :meth:`bigframes.session.Session.read_csv`
* :meth:`bigframes.dataframe.DataFrame.to_csv`

JSON
----

TODO: read_json

See also:

* :meth:`bigframes.dataframe.DataFrame.to_json`

Parquet
-------

TODO: read_parquet

See also:

* :meth:`bigframes.dataframe.DataFrame.to_parquet`

Google BigQuery
---------------

.. autofunction:: read_gbq
.. autofunction:: read_gbq_model

See also:

* :meth:`bigframes.session.Session.read_gbq`
* :meth:`bigframes.session.Session.read_gbq_model`
* :meth:`bigframes.dataframe.DataFrame.to_gbq`
