# Contains code from https://github.com/pola-rs/tpch/blob/main/queries/pandas/q7.py

from datetime import date
import typing

import bigframes
import bigframes.dataframe
import bigframes.pandas as bpd


def q(project_id: str, dataset_id: str, session: bigframes.Session):
    nation = session.read_gbq(
        f"{project_id}.{dataset_id}.NATION",
        index_col=bigframes.enums.DefaultIndexKind.NULL,
    )
    customer = session.read_gbq(
        f"{project_id}.{dataset_id}.CUSTOMER",
        index_col=bigframes.enums.DefaultIndexKind.NULL,
    )
    lineitem = session.read_gbq(
        f"{project_id}.{dataset_id}.LINEITEM",
        index_col=bigframes.enums.DefaultIndexKind.NULL,
    )
    orders = session.read_gbq(
        f"{project_id}.{dataset_id}.ORDERS",
        index_col=bigframes.enums.DefaultIndexKind.NULL,
    )
    supplier = session.read_gbq(
        f"{project_id}.{dataset_id}.SUPPLIER",
        index_col=bigframes.enums.DefaultIndexKind.NULL,
    )

    var1 = "FRANCE"
    var2 = "GERMANY"
    var3 = date(1995, 1, 1)
    var4 = date(1996, 12, 31)

    nation = nation[nation["N_NAME"].isin([var1, var2])]
    lineitem = lineitem[
        (lineitem["L_SHIPDATE"] >= var3) & (lineitem["L_SHIPDATE"] <= var4)
    ]

    jn1 = supplier.merge(nation, left_on="S_NATIONKEY", right_on="N_NATIONKEY")
    df1 = jn1.rename(columns={"N_NAME": "SUPP_NATION"})
    jn2 = lineitem.merge(df1, left_on="L_SUPPKEY", right_on="S_SUPPKEY")
    jn3 = orders.merge(jn2, left_on="O_ORDERKEY", right_on="L_ORDERKEY")
    jn4 = customer.merge(jn3, left_on="C_CUSTKEY", right_on="O_CUSTKEY")
    jn5 = jn4.merge(nation, left_on="C_NATIONKEY", right_on="N_NATIONKEY")
    jn5 = jn5.rename(columns={"N_NAME": "CUST_NATION"})
    total = jn5[jn5["CUST_NATION"] != jn5["SUPP_NATION"]]

    total["VOLUME"] = total["L_EXTENDEDPRICE"] * (1.0 - total["L_DISCOUNT"])
    total["L_YEAR"] = total["L_SHIPDATE"].dt.year

    gb = typing.cast(bpd.DataFrame, total).groupby(
        ["SUPP_NATION", "CUST_NATION", "L_YEAR"], as_index=False
    )
    agg = gb.agg(REVENUE=bpd.NamedAgg(column="VOLUME", aggfunc="sum"))

    result_df = typing.cast(bpd.DataFrame, agg).sort_values(
        ["SUPP_NATION", "CUST_NATION", "L_YEAR"]
    )
    next(result_df.to_pandas_batches(max_results=1500))
