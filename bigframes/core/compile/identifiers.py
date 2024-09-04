def map_id_to_sql(id: int) -> str:
    """Defines mapping between internal id type and labels to be used in ibis/sql"""
    return f"bfid_{id}"
