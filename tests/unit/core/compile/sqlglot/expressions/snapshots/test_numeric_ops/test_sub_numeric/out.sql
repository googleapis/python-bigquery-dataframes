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
  `int64_col` - `int64_col` AS `int_add_int`,
  `int64_col` - 1 AS `int_add_1`,
  `int64_col` - CAST(`bool_col` AS INT64) AS `int_add_bool`,
  CAST(`bool_col` AS INT64) - `int64_col` AS `bool_add_int`
FROM `bfcte_0`