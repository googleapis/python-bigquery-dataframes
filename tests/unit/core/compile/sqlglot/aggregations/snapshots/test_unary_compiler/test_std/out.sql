SELECT
  `bfcol_12` AS `int64_col`,
  `bfcol_13` AS `bool_col`,
  `bfcol_14` AS `duration_col`,
  `bfcol_15` AS `int64_col_w_floor`
FROM (
  SELECT
    STDDEV(`bfcol_6`) AS `bfcol_12`,
    STDDEV(CAST(`bfcol_7` AS INT64)) AS `bfcol_13`,
    CAST(FLOOR(STDDEV(`bfcol_8`)) AS INT64) AS `bfcol_14`,
    CAST(FLOOR(STDDEV(`bfcol_6`)) AS INT64) AS `bfcol_15`
  FROM `bigframes-dev`.`sqlglot_test`.`scalar_types` AS `bft_0`
)