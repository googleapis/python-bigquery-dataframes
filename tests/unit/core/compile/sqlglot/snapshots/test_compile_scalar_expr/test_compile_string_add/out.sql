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
), `bfcte_3` AS (
  SELECT
    `bfcol_4` AS `bfcol_7`,
    `bfcol_5` AS `bfcol_8`,
    `bfcol_6` AS `bfcol_9`
  FROM `bfcte_2`
)
SELECT
  `bfcol_7` AS `rowindex`,
  `bfcol_8` AS `string_col`
FROM `bfcte_3`