WITH `bfcte_1` AS (
  SELECT
    `rowindex` AS `bfcol_0`
  FROM `bigframes-dev`.`sqlglot_test`.`scalar_types`
), `bfcte_4` AS (
  SELECT
    `bfcol_0` AS `bfcol_1`
  FROM `bfcte_1`
), `bfcte_0` AS (
  SELECT
    `int64_col` AS `bfcol_2`,
    `rowindex` AS `bfcol_3`
  FROM `bigframes-dev`.`sqlglot_test`.`scalar_types`
), `bfcte_2` AS (
  SELECT
    *,
    `bfcol_3` AS `bfcol_6`,
    `bfcol_2` AS `bfcol_7`,
    `bfcol_2` AS `bfcol_8`
  FROM `bfcte_0`
), `bfcte_3` AS (
  SELECT
    *,
    `bfcol_7` + `bfcol_8` AS `bfcol_12`
  FROM `bfcte_2`
), `bfcte_5` AS (
  SELECT
    `bfcol_6` AS `bfcol_13`,
    `bfcol_12` AS `bfcol_14`
  FROM `bfcte_3`
), `bfcte_6` AS (
  SELECT
    *
  FROM `bfcte_4`
  LEFT JOIN `bfcte_5`
    ON COALESCE(`bfcol_1`, 0) = COALESCE(`bfcol_13`, 0)
    AND COALESCE(`bfcol_1`, 1) = COALESCE(`bfcol_13`, 1)
)
SELECT
  `bfcol_1` AS `rowindex`,
  `bfcol_14` AS `int64_col`
FROM `bfcte_6`
