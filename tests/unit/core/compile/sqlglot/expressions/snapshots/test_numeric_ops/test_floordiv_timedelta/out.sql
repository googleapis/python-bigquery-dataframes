WITH `bfcte_0` AS (
  SELECT
    `date_col`,
    `rowindex`,
    `timestamp_col`
  FROM `bigframes-dev`.`sqlglot_test`.`scalar_types`
)
SELECT
  `rowindex`,
  `timestamp_col`,
  `date_col`,
  43200000000 AS `timedelta_div_numeric`
FROM `bfcte_0`