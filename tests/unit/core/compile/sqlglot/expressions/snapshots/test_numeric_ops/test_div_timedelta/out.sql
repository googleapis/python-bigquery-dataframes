WITH `bfcte_0` AS (
  SELECT
    `int64_col`,
    `rowindex`,
    `timestamp_col`
  FROM `bigframes-dev`.`sqlglot_test`.`scalar_types`
)
SELECT
  `rowindex`,
  `timestamp_col`,
  `int64_col`,
  CAST(FLOOR(IEEE_DIVIDE(86400000000, `int64_col`)) AS INT64) AS `timedelta_div_numeric`
FROM `bfcte_0`