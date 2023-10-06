from typing import Mapping, Sequence, Tuple


class JoinNameRemapper:
    def __init__(self, namespace: str) -> None:
        self._namespace = namespace

    def __call__(
        self, left_column_ids: Sequence[str], right_column_ids: Sequence[str]
    ) -> Tuple[Mapping[str, str], Mapping[str, str]]:
        """When joining column ids from different namespaces, this function defines how names are remapped. Map only non-hidden ids."""
        # This naming strategy depends on the number of visible columns in source tables.
        # This means column id mappings must be adjusted if pushing operations above or below join in transformation
        new_left_ids = {
            col: f"{self._namespace}_l_{i}" for i, col in enumerate(left_column_ids)
        }
        new_right_ids = {
            col: f"{self._namespace}_r_{i}" for i, col in enumerate(right_column_ids)
        }
        return new_left_ids, new_right_ids


# Defines how column ids are remapped, regardless of join strategy or ordering mode
JOIN_NAME_REMAPPER = JoinNameRemapper("bfjoin")
