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
  `bool_col` AS `bool_col`,
  `int64_col` AS `int64_col`,
  `int64_col` & `int64_col` AS `int_and_int`,
  `bool_col` AND `bool_col` AS `bool_and_bool`,
  IF(`bool_col` = FALSE, `bool_col`, NULL) AS `bool_and_null`
FROM `bfcte_0`