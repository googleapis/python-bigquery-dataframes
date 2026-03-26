SELECT
  POWER(`t1`.`bfuid_col_1432`, `t1`.`bfuid_col_1432`) AS `int_pow_int`,
  POWER(`t1`.`bfuid_col_1432`, `t1`.`bfuid_col_1435`) AS `int_pow_float`,
  POWER(`t1`.`bfuid_col_1435`, `t1`.`bfuid_col_1432`) AS `float_pow_int`,
  POWER(`t1`.`bfuid_col_1435`, `t1`.`bfuid_col_1435`) AS `float_pow_float`,
  POWER(`t1`.`bfuid_col_1432`, CAST(`t1`.`bfuid_col_1427` AS INT64)) AS `int_pow_bool`,
  POWER(CAST(`t1`.`bfuid_col_1427` AS INT64), `t1`.`bfuid_col_1432`) AS `bool_pow_int`
FROM (
  SELECT
    `t0`.`bool_col` AS `bfuid_col_1427`,
    `t0`.`int64_col` AS `bfuid_col_1432`,
    `t0`.`float64_col` AS `bfuid_col_1435`,
    (
      `t0`.`int64_col` >= 0
    ) AND (
      `t0`.`int64_col` <= 10
    ) AS `bfuid_col_1442`
  FROM `bigframes-dev.sqlglot_test.scalar_types` AS `t0`
) AS `t1`
WHERE
  `t1`.`bfuid_col_1442`