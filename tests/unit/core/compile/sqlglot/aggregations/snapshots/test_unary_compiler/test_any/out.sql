SELECT
  `bfcol_2` AS `bool_col`,
  `bfcol_3` AS `int64_col`
FROM (
  SELECT
    COALESCE(LOGICAL_OR(`bool_col`), FALSE) AS `bfcol_2`,
    COALESCE(LOGICAL_OR(`int64_col` <> 0), FALSE) AS `bfcol_3`
  FROM `bigframes-dev`.`sqlglot_test`.`scalar_types` AS `bft_0`
)