SELECT
  SUBSTRING(
    `t0`.`string_col`,
    IF(
      (
        IF(1 >= 0, 1, GREATEST(0, LENGTH(`t0`.`string_col`) + 1)) + 1
      ) >= 1,
      IF(1 >= 0, 1, GREATEST(0, LENGTH(`t0`.`string_col`) + 1)) + 1,
      IF(1 >= 0, 1, GREATEST(0, LENGTH(`t0`.`string_col`) + 1)) + 1 + LENGTH(`t0`.`string_col`)
    ),
    GREATEST(
      0,
      IF(3 >= 0, 3, GREATEST(0, LENGTH(`t0`.`string_col`) + 3)) - IF(1 >= 0, 1, GREATEST(0, LENGTH(`t0`.`string_col`) + 1))
    )
  ) AS `string_col`
FROM (
  SELECT
    `string_col`
  FROM `bigframes-dev.sqlglot_test.scalar_types` FOR SYSTEM_TIME AS OF DATETIME('2025-09-30T20:19:48.854671')
) AS `t0`