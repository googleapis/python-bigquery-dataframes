SELECT
  CAST(timestamp_micros(
    CAST(trunc((
      `t0`.`rowindex` * 86400000000.0
    ) + UNIX_MICROS(`t0`.`timestamp_col`)) AS INT64)
  ) AS TIMESTAMP) AS `fixed_freq`
FROM `bigframes-dev.sqlglot_test.scalar_types` AS `t0`