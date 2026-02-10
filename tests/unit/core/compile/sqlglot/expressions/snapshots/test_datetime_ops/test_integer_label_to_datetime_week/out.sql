SELECT
  CAST(timestamp_micros(
    (
      `t0`.`rowindex` * 604800000000
    ) + UNIX_MICROS(
      TIMESTAMP_ADD(TIMESTAMP_TRUNC(`t0`.`timestamp_col`, WEEK(MONDAY)), INTERVAL '6' DAY)
    )
  ) AS TIMESTAMP) AS `non_fixed_freq_weekly`
FROM `bigframes-dev.sqlglot_test.scalar_types` AS `t0`