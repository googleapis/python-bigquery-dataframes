SELECT
  `bfcol_2` AS `cov_col`
FROM (
  SELECT
    COVAR_SAMP(`int64_col`, `float64_col`) AS `bfcol_2`
  FROM `bigframes-dev`.`sqlglot_test`.`scalar_types` AS `bft_0`
)