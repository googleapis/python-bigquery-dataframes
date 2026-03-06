SELECT
  `bfcol_4` AS `int64_col`,
  `bfcol_5` AS `bool_col`
FROM (
  SELECT
    VAR_POP(`int64_col`) AS `bfcol_4`,
    VAR_POP(CAST(`bool_col` AS INT64)) AS `bfcol_5`
  FROM `bigframes-dev`.`sqlglot_test`.`scalar_types` AS `bft_0`
)