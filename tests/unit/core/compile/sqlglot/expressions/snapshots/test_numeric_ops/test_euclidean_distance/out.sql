WITH `bfcte_0` AS (
  SELECT
    `int_list_col` AS `bfcol_0`,
    `numeric_list_col` AS `bfcol_1`
  FROM `bigframes-dev`.`sqlglot_test`.`repeated_types`
), `bfcte_1` AS (
  SELECT
    *,
    ML.DISTANCE(`bfcol_0`, `bfcol_0`, 'EUCLIDEAN') AS `bfcol_2`,
    ML.DISTANCE(`bfcol_1`, `bfcol_1`, 'EUCLIDEAN') AS `bfcol_3`
  FROM `bfcte_0`
)
SELECT
  `bfcol_2` AS `int_list_col`,
  `bfcol_3` AS `numeric_list_col`
FROM `bfcte_1`