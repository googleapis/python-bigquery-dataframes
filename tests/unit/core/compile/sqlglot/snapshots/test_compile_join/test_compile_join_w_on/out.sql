WITH `bfcte_1` AS (
  SELECT
    `int64_col` AS `bfcol_0`,
    `int64_too` AS `bfcol_1`,
    `rowindex` AS `bfcol_2`
  FROM `bigframes-dev`.`sqlglot_test`.`scalar_types`
), `bfcte_2` AS (
  SELECT
    `bfcol_1` AS `bfcol_3`,
    `bfcol_2` AS `bfcol_4`,
    `bfcol_0` AS `bfcol_5`
  FROM `bfcte_1`
), `bfcte_0` AS (
  SELECT
    `int64_col` AS `bfcol_6`,
    `int64_too` AS `bfcol_7`
  FROM `bigframes-dev`.`sqlglot_test`.`scalar_types`
), `bfcte_3` AS (
  SELECT
    `bfcol_7` AS `bfcol_8`,
    `bfcol_6` AS `bfcol_9`
  FROM `bfcte_0`
), `bfcte_4` AS (
  SELECT
    *
  FROM `bfcte_2`
  LEFT JOIN `bfcte_3`
    ON   `bfcte_2`.`bfcol_3` = `bfcte_3`.`bfcol_8`
)
SELECT
  `bfcol_4` AS `rowindex`,
  `bfcol_5` AS `int64_col`,
  `bfcol_3` AS `int64_too`,
  `bfcol_9` AS `col1`
FROM `bfcte_4`