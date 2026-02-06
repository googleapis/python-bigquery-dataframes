WITH `bfcte_0` AS (
  SELECT
    `bool_col`,
    `bytes_col`
  FROM `bigframes-dev`.`sqlglot_test`.`scalar_types`
)
SELECT
  CAST(`bool_col` AS INT64) + BYTE_LENGTH(`bytes_col`) AS `bool_col`
FROM `bfcte_0`