WITH `bfcte_0` AS (
  SELECT
    *
  FROM UNNEST(ARRAY<STRUCT<`bfcol_0` BOOLEAN, `bfcol_1` INT64>>[STRUCT(CAST(NULL AS BOOLEAN), CAST(NULL AS INT64))])
), `bfcte_1` AS (
  SELECT
    COALESCE(SUM(`bfcol_1`), 0) AS `bfcol_4`,
    COALESCE(SUM(CAST(`bfcol_0` AS INT64)), 0) AS `bfcol_5`
  FROM `bfcte_0`
)
SELECT
  `bfcol_4` AS `int64_col`,
  `bfcol_5` AS `bool_col`
FROM `bfcte_1`