WITH `bfcte_0` AS (
  SELECT
    `bool_col` AS `bfcol_0`,
    `rowindex` AS `bfcol_1`
  FROM `bigframes-dev`.`sqlglot_test`.`scalar_types`
), `bfcte_1` AS (
  SELECT
    *,
    NOT `bfcol_0` AS `bfcol_4`
  FROM `bfcte_0`
)
SELECT
  `bfcol_1` AS `rowindex`,
  `bfcol_4` AS `bool_col`
FROM `bfcte_1`