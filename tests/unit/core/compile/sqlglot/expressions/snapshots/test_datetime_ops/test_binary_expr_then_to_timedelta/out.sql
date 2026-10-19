SELECT
  `rowindex`,
  (
    `int64_col` + `int64_col`
  ) * 1000000 AS `int64_col`
FROM `bigframes-dev`.`sqlglot_test`.`scalar_types` AS `bft_0`