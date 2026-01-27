SELECT
  POWER(`t1`.`bfuid_col_1275`, `t1`.`bfuid_col_1275`) AS `int_pow_int`,
  POWER(`t1`.`bfuid_col_1275`, `t1`.`bfuid_col_1278`) AS `int_pow_float`,
  POWER(`t1`.`bfuid_col_1278`, `t1`.`bfuid_col_1275`) AS `float_pow_int`,
  POWER(`t1`.`bfuid_col_1278`, `t1`.`bfuid_col_1278`) AS `float_pow_float`,
  POWER(`t1`.`bfuid_col_1275`, CAST(`t1`.`bfuid_col_1270` AS INT64)) AS `int_pow_bool`,
  POWER(CAST(`t1`.`bfuid_col_1270` AS INT64), `t1`.`bfuid_col_1275`) AS `bool_pow_int`
FROM (
  SELECT
    `t0`.`bool_col` AS `bfuid_col_1270`,
    `t0`.`int64_col` AS `bfuid_col_1275`,
    `t0`.`float64_col` AS `bfuid_col_1278`,
    (
      `t0`.`int64_col` >= 0
    ) AND (
      `t0`.`int64_col` <= 10
    ) AS `bfuid_col_1285`
  FROM `bigframes-dev.sqlglot_test.scalar_types` AS `t0`
) AS `t1`
WHERE
  `t1`.`bfuid_col_1285`