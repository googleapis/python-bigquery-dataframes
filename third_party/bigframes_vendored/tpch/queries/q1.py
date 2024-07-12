# Contains code from https://github.com/pola-rs/tpch/blob/main/queries/pandas/q1.py

from datetime import datetime, timedelta

import bigframes.pandas as bpd


def q(dataset_id, session):

    lineitem = session.read_gbq(f"bigframes-dev-perf.{dataset_id}.lineitem")
    cutoff_date = datetime(1998, 12, 1) - timedelta(days=3)
    lineitem = lineitem[lineitem["L_SHIPDATE"] <= cutoff_date.date()]

    lineitem["DISC_PRICE"] = lineitem["L_EXTENDEDPRICE"] * (1 - lineitem["L_DISCOUNT"])
    lineitem["CHARGE_PRICE"] = lineitem["DISC_PRICE"] * (1 + lineitem["L_TAX"])

    result = lineitem.groupby(["L_RETURNFLAG", "L_LINESTATUS"]).agg(
        SUM_QTY=bpd.NamedAgg(column="L_QUANTITY", aggfunc="sum"),
        SUM_BASE_PRICE=bpd.NamedAgg(column="L_EXTENDEDPRICE", aggfunc="sum"),
        SUM_DISC_PRICE=bpd.NamedAgg(column="DISC_PRICE", aggfunc="sum"),
        SUM_CHARGE=bpd.NamedAgg(column="CHARGE_PRICE", aggfunc="sum"),
        AVG_QTY=bpd.NamedAgg(column="L_QUANTITY", aggfunc="mean"),
        AVG_PRICE=bpd.NamedAgg(column="L_EXTENDEDPRICE", aggfunc="mean"),
        AVG_DISC=bpd.NamedAgg(column="L_DISCOUNT", aggfunc="mean"),
        COUNT_ORDER=bpd.NamedAgg(column="L_QUANTITY", aggfunc="count"),
    )
    result = result.sort_index()

    print(result)

    bpd.reset_session()
