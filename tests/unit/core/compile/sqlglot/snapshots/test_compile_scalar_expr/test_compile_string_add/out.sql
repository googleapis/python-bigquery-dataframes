WITH `bfcte_0` AS (
  SELECT
    `rowindex` AS `bfcol_0`,
    `string_col` AS `bfcol_1`
  FROM `bigframes-dev`.`sqlglot_test`.`scalar_types`
), `bfcte_1` AS (
  SELECT
    `bfcol_0` AS `bfcol_2`,
    `bfcol_1` AS `bfcol_3`
  FROM `bfcte_0`
), `bfcte_2` AS (
  SELECT
    *,
    `bfcol_2` AS `bfcol_4`,
    `bfcol_3` AS `bfcol_5`,
    CONCAT(`bfcol_3`, 'a') AS `bfcol_6`
  FROM `bfcte_1`
)
SELECT
  `bfcol_4` AS `rowindex`,
  `bfcol_6` AS `string_col`
FROM `bfcte_2`