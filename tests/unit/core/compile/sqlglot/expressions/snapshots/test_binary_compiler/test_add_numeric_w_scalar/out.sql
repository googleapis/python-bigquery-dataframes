WITH `bfcte_0` AS (
  SELECT
    `int64_col` AS `bfcol_0`,
    `rowindex` AS `bfcol_1`
  FROM `bigframes-dev`.`sqlglot_test`.`scalar_types`
), `bfcte_1` AS (
  SELECT
    *,
    `bfcol_0` + 1 AS `bfcol_4`
  FROM `bfcte_0`
)
SELECT
  `bfcol_1` AS `bfuid_col_1`,
  `bfcol_0` AS `int64_col`,
  `bfcol_4` AS `bfuid_col_3`
FROM `bfcte_1`