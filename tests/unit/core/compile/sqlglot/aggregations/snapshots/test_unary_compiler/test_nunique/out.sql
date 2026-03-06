SELECT
  `bfcol_1` AS `int64_col`
FROM (
  SELECT
    COUNT(DISTINCT `int64_col`) AS `bfcol_1`
  FROM `bigframes-dev`.`sqlglot_test`.`scalar_types` AS `bft_0`
)