_GUID_COUNTER = 0


def generate_guid(prefix="col_"):
    global _GUID_COUNTER
    _GUID_COUNTER += 1
    return prefix + str(_GUID_COUNTER)
