SELECT
  `bfcol_1` AS `int64_col`
FROM (
  SELECT
    ANY_VALUE(`int64_col`) AS `bfcol_1`
  FROM (
    SELECT
      `int64_col`
    FROM `bigframes-dev`.`sqlglot_test`.`scalar_types` AS `bft_0`
  )
)