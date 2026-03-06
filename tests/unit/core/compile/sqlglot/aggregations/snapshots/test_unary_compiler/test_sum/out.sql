SELECT
  `bfcol_4` AS `int64_col`,
  `bfcol_5` AS `bool_col`
FROM (
  SELECT
    COALESCE(SUM(`int64_col`), 0) AS `bfcol_4`,
    COALESCE(SUM(CAST(`bool_col` AS INT64)), 0) AS `bfcol_5`
  FROM `bigframes-dev`.`sqlglot_test`.`scalar_types` AS `bft_0`
)