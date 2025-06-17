WITH `bfcte_0` AS (
  SELECT
    `int64_col` AS `bfcol_0`,
    `rowindex` AS `bfcol_1`
  FROM `bigframes-dev`.`sqlglot_test`.`scalar_types`
), `bfcte_1` AS (
  SELECT
    `bfcol_1` AS `bfcol_2`,
    `bfcol_0` AS `bfcol_3`
  FROM `bfcte_0`
), `bfcte_2` AS (
  SELECT
    *,
    `bfcol_2` AS `bfcol_4`,
    `bfcol_3` AS `bfcol_5`,
    `bfcol_3` + `bfcol_3` AS `bfcol_6`
  FROM `bfcte_1`
)
SELECT
  `bfcol_4` AS `rowindex`,
  `bfcol_6` AS `int64_col`
FROM `bfcte_2`