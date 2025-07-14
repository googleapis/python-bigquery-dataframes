WITH `bfcte_0` AS (
  SELECT
    `int64_col` AS `bfcol_0`,
    `rowindex` AS `bfcol_1`
  FROM `bigframes-dev`.`sqlglot_test`.`scalar_types`
), `bfcte_1` AS (
  SELECT
    *,
    IF(ABS(`bfcol_0`) <= 1, ACOS(`bfcol_0`), CAST('NaN' AS FLOAT64)) AS `bfcol_4`
  FROM `bfcte_0`
)
SELECT
  `bfcol_1` AS `rowindex`,
  `bfcol_4` AS `int64_col`
FROM `bfcte_1`