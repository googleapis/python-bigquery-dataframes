SELECT
  `t0`.`rowindex`,
  `t0`.`int64_col`,
  `t0`.`bool_col`,
  `t0`.`float64_col`,
  CASE
    WHEN `t0`.`int64_col` = 0
    THEN 0 * `t0`.`int64_col`
    ELSE CAST(FLOOR(ieee_divide(`t0`.`int64_col`, `t0`.`int64_col`)) AS INT64)
  END AS `int_div_int`,
  CASE
    WHEN 1 = 0
    THEN 0 * `t0`.`int64_col`
    ELSE CAST(FLOOR(ieee_divide(`t0`.`int64_col`, 1)) AS INT64)
  END AS `int_div_1`,
  CASE
    WHEN 0.0 = 0
    THEN CAST('Infinity' AS FLOAT64) * `t0`.`int64_col`
    ELSE CAST(FLOOR(ieee_divide(`t0`.`int64_col`, 0.0)) AS INT64)
  END AS `int_div_0`,
  CAST(NULL AS INT64) AS `int_div_null`,
  CASE
    WHEN `t0`.`float64_col` = 0
    THEN CAST('Infinity' AS FLOAT64) * `t0`.`int64_col`
    ELSE CAST(FLOOR(ieee_divide(`t0`.`int64_col`, `t0`.`float64_col`)) AS INT64)
  END AS `int_div_float`,
  CASE
    WHEN `t0`.`int64_col` = 0
    THEN CAST('Infinity' AS FLOAT64) * `t0`.`float64_col`
    ELSE CAST(FLOOR(ieee_divide(`t0`.`float64_col`, `t0`.`int64_col`)) AS INT64)
  END AS `float_div_int`,
  CASE
    WHEN 0.0 = 0
    THEN CAST('Infinity' AS FLOAT64) * `t0`.`float64_col`
    ELSE CAST(FLOOR(ieee_divide(`t0`.`float64_col`, 0.0)) AS INT64)
  END AS `float_div_0`,
  CASE
    WHEN CAST(`t0`.`bool_col` AS INT64) = 0
    THEN 0 * `t0`.`int64_col`
    ELSE CAST(FLOOR(ieee_divide(`t0`.`int64_col`, CAST(`t0`.`bool_col` AS INT64))) AS INT64)
  END AS `int_div_bool`,
  CASE
    WHEN `t0`.`int64_col` = 0
    THEN 0 * CAST(`t0`.`bool_col` AS INT64)
    ELSE CAST(FLOOR(ieee_divide(CAST(`t0`.`bool_col` AS INT64), `t0`.`int64_col`)) AS INT64)
  END AS `bool_div_int`
FROM `bigframes-dev.sqlglot_test.scalar_types` AS `t0`