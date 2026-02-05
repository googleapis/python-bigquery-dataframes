WITH `bfcte_0` AS (
  SELECT
    `bool_col`,
    `int64_col`,
    `rowindex`
  FROM `bigframes-dev`.`sqlglot_test`.`scalar_types`
)
SELECT
  *,
  `rowindex` AS `rowindex`,
  `int64_col` AS `int64_col`,
  `bool_col` AS `bool_col`,
  `int64_col` = `int64_col` AS `int_eq_int`,
  `int64_col` = 1 AS `int_eq_1`,
  `int64_col` IS NULL AS `int_eq_null`,
  `int64_col` = CAST(`bool_col` AS INT64) AS `int_eq_bool`,
  CAST(`bool_col` AS INT64) = `int64_col` AS `bool_eq_int`
FROM `bfcte_0`