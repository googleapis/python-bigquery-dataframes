SELECT
  `t0`.`rowindex`,
  `t0`.`int64_col`,
  `t0`.`float64_col`,
  CASE
    WHEN `t0`.`int64_col` = 0
    THEN 0 * `t0`.`int64_col`
    WHEN (
      `t0`.`int64_col` < 0
    )
    AND (
      (
        MOD(`t0`.`int64_col`, `t0`.`int64_col`)
      ) > 0
    )
    THEN `t0`.`int64_col` + (
      MOD(`t0`.`int64_col`, `t0`.`int64_col`)
    )
    WHEN (
      `t0`.`int64_col` > 0
    )
    AND (
      (
        MOD(`t0`.`int64_col`, `t0`.`int64_col`)
      ) < 0
    )
    THEN `t0`.`int64_col` + (
      MOD(`t0`.`int64_col`, `t0`.`int64_col`)
    )
    ELSE MOD(`t0`.`int64_col`, `t0`.`int64_col`)
  END AS `int_mod_int`,
  CASE
    WHEN -(
      `t0`.`int64_col`
    ) = 0
    THEN 0 * `t0`.`int64_col`
    WHEN (
      -(
        `t0`.`int64_col`
      ) < 0
    )
    AND (
      (
        MOD(`t0`.`int64_col`, -(
          `t0`.`int64_col`
        ))
      ) > 0
    )
    THEN -(
      `t0`.`int64_col`
    ) + (
      MOD(`t0`.`int64_col`, -(
        `t0`.`int64_col`
      ))
    )
    WHEN (
      -(
        `t0`.`int64_col`
      ) > 0
    )
    AND (
      (
        MOD(`t0`.`int64_col`, -(
          `t0`.`int64_col`
        ))
      ) < 0
    )
    THEN -(
      `t0`.`int64_col`
    ) + (
      MOD(`t0`.`int64_col`, -(
        `t0`.`int64_col`
      ))
    )
    ELSE MOD(`t0`.`int64_col`, -(
      `t0`.`int64_col`
    ))
  END AS `int_mod_int_neg`,
  CASE
    WHEN 1 = 0
    THEN 0 * `t0`.`int64_col`
    WHEN (
      1 < 0
    ) AND (
      (
        MOD(`t0`.`int64_col`, 1)
      ) > 0
    )
    THEN 1 + (
      MOD(`t0`.`int64_col`, 1)
    )
    WHEN (
      1 > 0
    ) AND (
      (
        MOD(`t0`.`int64_col`, 1)
      ) < 0
    )
    THEN 1 + (
      MOD(`t0`.`int64_col`, 1)
    )
    ELSE MOD(`t0`.`int64_col`, 1)
  END AS `int_mod_1`,
  CAST(NULL AS INT64) AS `int_mod_0`,
  CAST(CASE
    WHEN CAST(`t0`.`float64_col` AS BIGNUMERIC) = 0
    THEN CAST('NaN' AS FLOAT64) * CAST(`t0`.`float64_col` AS BIGNUMERIC)
    WHEN (
      CAST(`t0`.`float64_col` AS BIGNUMERIC) < 0
    )
    AND (
      (
        MOD(CAST(`t0`.`float64_col` AS BIGNUMERIC), CAST(`t0`.`float64_col` AS BIGNUMERIC))
      ) > 0
    )
    THEN CAST(`t0`.`float64_col` AS BIGNUMERIC) + (
      MOD(CAST(`t0`.`float64_col` AS BIGNUMERIC), CAST(`t0`.`float64_col` AS BIGNUMERIC))
    )
    WHEN (
      CAST(`t0`.`float64_col` AS BIGNUMERIC) > 0
    )
    AND (
      (
        MOD(CAST(`t0`.`float64_col` AS BIGNUMERIC), CAST(`t0`.`float64_col` AS BIGNUMERIC))
      ) < 0
    )
    THEN CAST(`t0`.`float64_col` AS BIGNUMERIC) + (
      MOD(CAST(`t0`.`float64_col` AS BIGNUMERIC), CAST(`t0`.`float64_col` AS BIGNUMERIC))
    )
    ELSE MOD(CAST(`t0`.`float64_col` AS BIGNUMERIC), CAST(`t0`.`float64_col` AS BIGNUMERIC))
  END AS FLOAT64) AS `float_mod_float`,
  CAST(CASE
    WHEN CAST(-(
      `t0`.`float64_col`
    ) AS BIGNUMERIC) = 0
    THEN CAST('NaN' AS FLOAT64) * CAST(`t0`.`float64_col` AS BIGNUMERIC)
    WHEN (
      CAST(-(
        `t0`.`float64_col`
      ) AS BIGNUMERIC) < 0
    )
    AND (
      (
        MOD(
          CAST(`t0`.`float64_col` AS BIGNUMERIC),
          CAST(-(
            `t0`.`float64_col`
          ) AS BIGNUMERIC)
        )
      ) > 0
    )
    THEN CAST(-(
      `t0`.`float64_col`
    ) AS BIGNUMERIC) + (
      MOD(
        CAST(`t0`.`float64_col` AS BIGNUMERIC),
        CAST(-(
          `t0`.`float64_col`
        ) AS BIGNUMERIC)
      )
    )
    WHEN (
      CAST(-(
        `t0`.`float64_col`
      ) AS BIGNUMERIC) > 0
    )
    AND (
      (
        MOD(
          CAST(`t0`.`float64_col` AS BIGNUMERIC),
          CAST(-(
            `t0`.`float64_col`
          ) AS BIGNUMERIC)
        )
      ) < 0
    )
    THEN CAST(-(
      `t0`.`float64_col`
    ) AS BIGNUMERIC) + (
      MOD(
        CAST(`t0`.`float64_col` AS BIGNUMERIC),
        CAST(-(
          `t0`.`float64_col`
        ) AS BIGNUMERIC)
      )
    )
    ELSE MOD(
      CAST(`t0`.`float64_col` AS BIGNUMERIC),
      CAST(-(
        `t0`.`float64_col`
      ) AS BIGNUMERIC)
    )
  END AS FLOAT64) AS `float_mod_float_neg`,
  CAST(CASE
    WHEN CAST(1 AS BIGNUMERIC) = 0
    THEN CAST('NaN' AS FLOAT64) * CAST(`t0`.`float64_col` AS BIGNUMERIC)
    WHEN (
      CAST(1 AS BIGNUMERIC) < 0
    )
    AND (
      (
        MOD(CAST(`t0`.`float64_col` AS BIGNUMERIC), CAST(1 AS BIGNUMERIC))
      ) > 0
    )
    THEN CAST(1 AS BIGNUMERIC) + (
      MOD(CAST(`t0`.`float64_col` AS BIGNUMERIC), CAST(1 AS BIGNUMERIC))
    )
    WHEN (
      CAST(1 AS BIGNUMERIC) > 0
    )
    AND (
      (
        MOD(CAST(`t0`.`float64_col` AS BIGNUMERIC), CAST(1 AS BIGNUMERIC))
      ) < 0
    )
    THEN CAST(1 AS BIGNUMERIC) + (
      MOD(CAST(`t0`.`float64_col` AS BIGNUMERIC), CAST(1 AS BIGNUMERIC))
    )
    ELSE MOD(CAST(`t0`.`float64_col` AS BIGNUMERIC), CAST(1 AS BIGNUMERIC))
  END AS FLOAT64) AS `float_mod_1`,
  CAST(NULL AS FLOAT64) AS `float_mod_0`
FROM `bigframes-dev.sqlglot_test.scalar_types` AS `t0`