SELECT
  `t0`.`rowindex`,
  `t0`.`int64_col`,
  `t0`.`float64_col`,
  CASE
    WHEN (
      (
        POWER(CAST(`t0`.`int64_col` AS NUMERIC), `t0`.`int64_col`)
      ) > 9223372036854775807
    )
    OR (
      (
        POWER(CAST(`t0`.`int64_col` AS NUMERIC), `t0`.`int64_col`)
      ) < -9223372036854775808
    )
    THEN NULL
    ELSE CAST(trunc(POWER(CAST(`t0`.`int64_col` AS NUMERIC), `t0`.`int64_col`)) AS INT64)
  END AS `int_pow_int`,
  CASE
    WHEN `t0`.`float64_col` = 0
    THEN 1
    WHEN `t0`.`int64_col` = 1
    THEN 1
    WHEN (
      `t0`.`int64_col` = 0
    ) AND (
      `t0`.`float64_col` < 0
    )
    THEN CAST('Infinity' AS FLOAT64)
    WHEN ABS(`t0`.`int64_col`) = CAST('Infinity' AS FLOAT64)
    THEN POWER(
      `t0`.`int64_col`,
      IF(
        ABS(`t0`.`float64_col`) > 9007199254740992,
        CAST('Infinity' AS FLOAT64) * SIGN(`t0`.`float64_col`),
        `t0`.`float64_col`
      )
    )
    WHEN ABS(`t0`.`float64_col`) > 9007199254740992
    THEN POWER(
      `t0`.`int64_col`,
      IF(
        ABS(`t0`.`float64_col`) > 9007199254740992,
        CAST('Infinity' AS FLOAT64) * SIGN(`t0`.`float64_col`),
        `t0`.`float64_col`
      )
    )
    WHEN (
      `t0`.`int64_col` < 0
    )
    AND NOT (
      CAST(trunc(`t0`.`float64_col`) AS INT64) = `t0`.`float64_col`
    )
    THEN CAST('NaN' AS FLOAT64)
    WHEN (
      `t0`.`int64_col` <> 0
    )
    AND (
      (
        `t0`.`float64_col` * LN(ABS(`t0`.`int64_col`))
      ) > 709.78
    )
    THEN CAST('Infinity' AS FLOAT64) * IF(
      (
        `t0`.`int64_col` < 0
      )
      AND (
        (
          MOD(CAST(trunc(`t0`.`float64_col`) AS INT64), 2)
        ) = 1
      ),
      -1,
      1
    )
    ELSE POWER(
      `t0`.`int64_col`,
      IF(
        ABS(`t0`.`float64_col`) > 9007199254740992,
        CAST('Infinity' AS FLOAT64) * SIGN(`t0`.`float64_col`),
        `t0`.`float64_col`
      )
    )
  END AS `int_pow_float`,
  CASE
    WHEN `t0`.`int64_col` = 0
    THEN 1
    WHEN `t0`.`float64_col` = 1
    THEN 1
    WHEN (
      `t0`.`float64_col` = 0
    ) AND (
      `t0`.`int64_col` < 0
    )
    THEN CAST('Infinity' AS FLOAT64)
    WHEN ABS(`t0`.`float64_col`) = CAST('Infinity' AS FLOAT64)
    THEN POWER(
      `t0`.`float64_col`,
      IF(
        ABS(`t0`.`int64_col`) > 9007199254740992,
        CAST('Infinity' AS FLOAT64) * SIGN(`t0`.`int64_col`),
        `t0`.`int64_col`
      )
    )
    WHEN ABS(`t0`.`int64_col`) > 9007199254740992
    THEN POWER(
      `t0`.`float64_col`,
      IF(
        ABS(`t0`.`int64_col`) > 9007199254740992,
        CAST('Infinity' AS FLOAT64) * SIGN(`t0`.`int64_col`),
        `t0`.`int64_col`
      )
    )
    WHEN (
      `t0`.`float64_col` < 0
    ) AND NOT (
      `t0`.`int64_col` = `t0`.`int64_col`
    )
    THEN CAST('NaN' AS FLOAT64)
    WHEN (
      `t0`.`float64_col` <> 0
    )
    AND (
      (
        `t0`.`int64_col` * LN(ABS(`t0`.`float64_col`))
      ) > 709.78
    )
    THEN CAST('Infinity' AS FLOAT64) * IF((
      `t0`.`float64_col` < 0
    ) AND (
      (
        MOD(`t0`.`int64_col`, 2)
      ) = 1
    ), -1, 1)
    ELSE POWER(
      `t0`.`float64_col`,
      IF(
        ABS(`t0`.`int64_col`) > 9007199254740992,
        CAST('Infinity' AS FLOAT64) * SIGN(`t0`.`int64_col`),
        `t0`.`int64_col`
      )
    )
  END AS `float_pow_int`,
  CASE
    WHEN `t0`.`float64_col` = 0
    THEN 1
    WHEN `t0`.`float64_col` = 1
    THEN 1
    WHEN (
      `t0`.`float64_col` = 0
    ) AND (
      `t0`.`float64_col` < 0
    )
    THEN CAST('Infinity' AS FLOAT64)
    WHEN ABS(`t0`.`float64_col`) = CAST('Infinity' AS FLOAT64)
    THEN POWER(
      `t0`.`float64_col`,
      IF(
        ABS(`t0`.`float64_col`) > 9007199254740992,
        CAST('Infinity' AS FLOAT64) * SIGN(`t0`.`float64_col`),
        `t0`.`float64_col`
      )
    )
    WHEN ABS(`t0`.`float64_col`) > 9007199254740992
    THEN POWER(
      `t0`.`float64_col`,
      IF(
        ABS(`t0`.`float64_col`) > 9007199254740992,
        CAST('Infinity' AS FLOAT64) * SIGN(`t0`.`float64_col`),
        `t0`.`float64_col`
      )
    )
    WHEN (
      `t0`.`float64_col` < 0
    )
    AND NOT (
      CAST(trunc(`t0`.`float64_col`) AS INT64) = `t0`.`float64_col`
    )
    THEN CAST('NaN' AS FLOAT64)
    WHEN (
      `t0`.`float64_col` <> 0
    )
    AND (
      (
        `t0`.`float64_col` * LN(ABS(`t0`.`float64_col`))
      ) > 709.78
    )
    THEN CAST('Infinity' AS FLOAT64) * IF(
      (
        `t0`.`float64_col` < 0
      )
      AND (
        (
          MOD(CAST(trunc(`t0`.`float64_col`) AS INT64), 2)
        ) = 1
      ),
      -1,
      1
    )
    ELSE POWER(
      `t0`.`float64_col`,
      IF(
        ABS(`t0`.`float64_col`) > 9007199254740992,
        CAST('Infinity' AS FLOAT64) * SIGN(`t0`.`float64_col`),
        `t0`.`float64_col`
      )
    )
  END AS `float_pow_float`,
  CASE
    WHEN (
      (
        POWER(CAST(`t0`.`int64_col` AS NUMERIC), 0)
      ) > 9223372036854775807
    )
    OR (
      (
        POWER(CAST(`t0`.`int64_col` AS NUMERIC), 0)
      ) < -9223372036854775808
    )
    THEN NULL
    ELSE CAST(trunc(POWER(CAST(`t0`.`int64_col` AS NUMERIC), 0)) AS INT64)
  END AS `int_pow_0`,
  CASE
    WHEN 0 = 0
    THEN 1
    WHEN `t0`.`float64_col` = 1
    THEN 1
    WHEN (
      `t0`.`float64_col` = 0
    ) AND (
      0 < 0
    )
    THEN CAST('Infinity' AS FLOAT64)
    WHEN ABS(`t0`.`float64_col`) = CAST('Infinity' AS FLOAT64)
    THEN POWER(
      `t0`.`float64_col`,
      IF(ABS(0) > 9007199254740992, CAST('Infinity' AS FLOAT64) * SIGN(0), 0)
    )
    WHEN ABS(0) > 9007199254740992
    THEN POWER(
      `t0`.`float64_col`,
      IF(ABS(0) > 9007199254740992, CAST('Infinity' AS FLOAT64) * SIGN(0), 0)
    )
    WHEN (
      `t0`.`float64_col` < 0
    ) AND NOT (
      0 = 0
    )
    THEN CAST('NaN' AS FLOAT64)
    WHEN (
      `t0`.`float64_col` <> 0
    )
    AND (
      (
        0 * LN(ABS(`t0`.`float64_col`))
      ) > 709.78
    )
    THEN CAST('Infinity' AS FLOAT64) * IF((
      `t0`.`float64_col` < 0
    ) AND (
      (
        MOD(0, 2)
      ) = 1
    ), -1, 1)
    ELSE POWER(
      `t0`.`float64_col`,
      IF(ABS(0) > 9007199254740992, CAST('Infinity' AS FLOAT64) * SIGN(0), 0)
    )
  END AS `float_pow_0`,
  CASE
    WHEN (
      (
        POWER(CAST(`t0`.`int64_col` AS NUMERIC), 1)
      ) > 9223372036854775807
    )
    OR (
      (
        POWER(CAST(`t0`.`int64_col` AS NUMERIC), 1)
      ) < -9223372036854775808
    )
    THEN NULL
    ELSE CAST(trunc(POWER(CAST(`t0`.`int64_col` AS NUMERIC), 1)) AS INT64)
  END AS `int_pow_1`,
  CASE
    WHEN 1 = 0
    THEN 1
    WHEN `t0`.`float64_col` = 1
    THEN 1
    WHEN (
      `t0`.`float64_col` = 0
    ) AND (
      1 < 0
    )
    THEN CAST('Infinity' AS FLOAT64)
    WHEN ABS(`t0`.`float64_col`) = CAST('Infinity' AS FLOAT64)
    THEN POWER(
      `t0`.`float64_col`,
      IF(ABS(1) > 9007199254740992, CAST('Infinity' AS FLOAT64) * SIGN(1), 1)
    )
    WHEN ABS(1) > 9007199254740992
    THEN POWER(
      `t0`.`float64_col`,
      IF(ABS(1) > 9007199254740992, CAST('Infinity' AS FLOAT64) * SIGN(1), 1)
    )
    WHEN (
      `t0`.`float64_col` < 0
    ) AND NOT (
      1 = 1
    )
    THEN CAST('NaN' AS FLOAT64)
    WHEN (
      `t0`.`float64_col` <> 0
    )
    AND (
      (
        1 * LN(ABS(`t0`.`float64_col`))
      ) > 709.78
    )
    THEN CAST('Infinity' AS FLOAT64) * IF((
      `t0`.`float64_col` < 0
    ) AND (
      (
        MOD(1, 2)
      ) = 1
    ), -1, 1)
    ELSE POWER(
      `t0`.`float64_col`,
      IF(ABS(1) > 9007199254740992, CAST('Infinity' AS FLOAT64) * SIGN(1), 1)
    )
  END AS `float_pow_1`
FROM `bigframes-dev.sqlglot_test.scalar_types` AS `t0`