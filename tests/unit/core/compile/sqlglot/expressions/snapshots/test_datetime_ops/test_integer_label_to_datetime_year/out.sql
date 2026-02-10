SELECT
  CAST(TIMESTAMP_SUB(
    DATETIME(
      (
        (
          `t0`.`rowindex` * 1
        ) + EXTRACT(year FROM `t0`.`timestamp_col`)
      ) + 1,
      1,
      1,
      0,
      0,
      0
    ),
    INTERVAL '1' DAY
  ) AS TIMESTAMP) AS `non_fixed_freq_yearly`
FROM `bigframes-dev.sqlglot_test.scalar_types` AS `t0`