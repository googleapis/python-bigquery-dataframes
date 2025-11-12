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
FROM `bigframes-dev.sqlglot_test.scalar_types` AS `t0`