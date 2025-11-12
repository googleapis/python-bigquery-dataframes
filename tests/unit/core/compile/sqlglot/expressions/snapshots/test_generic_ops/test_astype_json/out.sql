SELECT
  PARSE_JSON(CAST(`t1`.`int64_col` AS STRING)) AS `int64_col`,
  PARSE_JSON(CAST(`t1`.`float64_col` AS STRING)) AS `float64_col`,
  PARSE_JSON(
    LOWER(
      CONCAT(
        UPPER(
          SUBSTRING(
            CAST(`t1`.`bool_col` AS STRING),
            IF((
              0 + 1
            ) >= 1, 0 + 1, 0 + 1 + LENGTH(CAST(`t1`.`bool_col` AS STRING))),
            1
          )
        ),
        LOWER(
          SUBSTRING(
            CAST(`t1`.`bool_col` AS STRING),
            IF((
              1 + 1
            ) >= 1, 1 + 1, 1 + 1 + LENGTH(CAST(`t1`.`bool_col` AS STRING))),
            LENGTH(CAST(`t1`.`bool_col` AS STRING))
          )
        )
      )
    )
  ) AS `bool_col`,
  PARSE_JSON(`t1`.`string_col`) AS `string_col`,
  SAFE.PARSE_JSON(
    LOWER(
      CONCAT(
        UPPER(
          SUBSTRING(
            SAFE_CAST(`t1`.`bool_col` AS STRING),
            IF((
              0 + 1
            ) >= 1, 0 + 1, 0 + 1 + LENGTH(SAFE_CAST(`t1`.`bool_col` AS STRING))),
            1
          )
        ),
        LOWER(
          SUBSTRING(
            SAFE_CAST(`t1`.`bool_col` AS STRING),
            IF((
              1 + 1
            ) >= 1, 1 + 1, 1 + 1 + LENGTH(SAFE_CAST(`t1`.`bool_col` AS STRING))),
            LENGTH(SAFE_CAST(`t1`.`bool_col` AS STRING))
          )
        )
      )
    )
  ) AS `bool_w_safe`,
  SAFE.PARSE_JSON(`t1`.`string_col`) AS `string_w_safe`
FROM (
  SELECT
    `t0`.`bool_col`,
    `t0`.`int64_col`,
    `t0`.`float64_col`,
    `t0`.`string_col`
  FROM `bigframes-dev.sqlglot_test.scalar_types` AS `t0`
) AS `t1`