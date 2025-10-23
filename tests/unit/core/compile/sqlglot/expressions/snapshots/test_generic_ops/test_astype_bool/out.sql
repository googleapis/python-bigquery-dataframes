WITH `bfcte_0` AS (
  SELECT
    `bool_col` AS `bfcol_0`,
    `float64_col` AS `bfcol_1`
  FROM `bigframes-dev`.`sqlglot_test`.`scalar_types`
), `bfcte_1` AS (
  SELECT
    *,
    `bfcol_0` AS `bfcol_2`,
    `bfcol_1` <> 0 AS `bfcol_3`,
    `bfcol_1` <> 0 AS `bfcol_4`
  FROM `bfcte_0`
)
SELECT
  `bfcol_2` AS `bool_col`,
  `bfcol_3` AS `float64_col`,
  `bfcol_4` AS `float64_w_safe`
FROM `bfcte_1`