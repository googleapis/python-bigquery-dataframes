WITH `bfcte_0` AS (
  SELECT
    `duration_col`,
    `int64_col`,
    `rowindex`,
    `timestamp_col`
  FROM `bigframes-dev`.`sqlglot_test`.`scalar_types`
)
SELECT
  *,
  `rowindex` AS `rowindex`,
  `timestamp_col` AS `timestamp_col`,
  `int64_col` AS `int64_col`,
  `duration_col` AS `duration_col`,
  CAST(FLOOR(`duration_col` * `int64_col`) AS INT64) AS `timedelta_mul_numeric`,
  CAST(FLOOR(`int64_col` * `duration_col`) AS INT64) AS `numeric_mul_timedelta`
FROM `bfcte_0`