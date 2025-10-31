WITH `bfcte_0` AS (
  SELECT
    `int64_col` AS `bfcol_0`,
    `int64_too` AS `bfcol_1`
  FROM `bigframes-dev`.`sqlglot_test`.`scalar_types`
), `bfcte_1` AS (
  SELECT
    *,
    COALESCE(`bfcol_0`, `bfcol_1`) AS `bfcol_2`
  FROM `bfcte_0`
)
SELECT
  `bfcol_2` AS `int64_col`
FROM `bfcte_1`