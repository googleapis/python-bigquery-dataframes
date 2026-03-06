SELECT
  `bfcol_1` AS `int64_col`
FROM (
  SELECT
    ARRAY_AGG(`int64_col` IGNORE NULLS ORDER BY `int64_col` IS NULL ASC, `int64_col` ASC) AS `bfcol_1`
  FROM `bigframes-dev`.`sqlglot_test`.`scalar_types` AS `bft_0`
)