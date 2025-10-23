WITH `bfcte_0` AS (
  SELECT
    *
  FROM UNNEST(ARRAY<STRUCT<`bfcol_0` BOOLEAN, `bfcol_1` BYTES, `bfcol_2` INT64, `bfcol_3` INT64>>[STRUCT(CAST(NULL AS BOOLEAN), CAST(NULL AS BYTES), CAST(NULL AS INT64), 0)])
), `bfcte_1` AS (
  SELECT
    *,
    ~`bfcol_2` AS `bfcol_8`,
    ~`bfcol_1` AS `bfcol_9`,
    NOT `bfcol_0` AS `bfcol_10`
  FROM `bfcte_0`
)
SELECT
  `bfcol_8` AS `int64_col`,
  `bfcol_9` AS `bytes_col`,
  `bfcol_10` AS `bool_col`
FROM `bfcte_1`
ORDER BY
  `bfcol_3` ASC NULLS LAST