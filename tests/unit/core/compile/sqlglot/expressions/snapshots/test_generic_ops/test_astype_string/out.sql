SELECT
  CAST(`t0`.`int64_col` AS STRING) AS `int64_col`,
  CONCAT(
    UPPER(
      SUBSTRING(
        CAST(`t0`.`bool_col` AS STRING),
        IF((
          0 + 1
        ) >= 1, 0 + 1, 0 + 1 + LENGTH(CAST(`t0`.`bool_col` AS STRING))),
        1
      )
    ),
    LOWER(
      SUBSTRING(
        CAST(`t0`.`bool_col` AS STRING),
        IF((
          1 + 1
        ) >= 1, 1 + 1, 1 + 1 + LENGTH(CAST(`t0`.`bool_col` AS STRING))),
        LENGTH(CAST(`t0`.`bool_col` AS STRING))
      )
    )
  ) AS `bool_col`,
  CONCAT(
    UPPER(
      SUBSTRING(
        SAFE_CAST(`t0`.`bool_col` AS STRING),
        IF((
          0 + 1
        ) >= 1, 0 + 1, 0 + 1 + LENGTH(SAFE_CAST(`t0`.`bool_col` AS STRING))),
        1
      )
    ),
    LOWER(
      SUBSTRING(
        SAFE_CAST(`t0`.`bool_col` AS STRING),
        IF((
          1 + 1
        ) >= 1, 1 + 1, 1 + 1 + LENGTH(SAFE_CAST(`t0`.`bool_col` AS STRING))),
        LENGTH(SAFE_CAST(`t0`.`bool_col` AS STRING))
      )
    )
  ) AS `bool_w_safe`
FROM (
  SELECT
    `bool_col`,
    `int64_col`
  FROM `bigframes-dev.sqlglot_test.scalar_types` FOR SYSTEM_TIME AS OF DATETIME('2025-09-30T20:19:48.854671')
) AS `t0`