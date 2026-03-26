SELECT `PROMO_REVENUE` FROM (SELECT
  ROUND(ieee_divide(`t30`.`bfuid_col_3395`, `t32`.`bfuid_col_3387`), 2) AS `PROMO_REVENUE`,
  COALESCE(`t30`.`bfuid_col_3391`, `t32`.`bfuid_col_3384`) AS `bfuid_col_3403`
FROM (
  SELECT
    `t27`.`bfuid_col_3391`,
    100.0 * `t27`.`bfuid_col_3394` AS `bfuid_col_3395`
  FROM (
    SELECT
      `t25`.`bfuid_col_3391`,
      `t25`.`bfuid_col_3392`,
      ANY_VALUE(`t25`.`bfuid_col_3393`) AS `bfuid_col_3394`
    FROM (
      SELECT
        `t10`.`col_0` AS `bfuid_col_3391`,
        `t10`.`col_1` AS `bfuid_col_3392`,
        CASE
          WHEN `t22`.`bfuid_col_3389` = 0
          THEN CASE
            WHEN `t10`.`col_2` = 0
            THEN `t22`.`bfuid_col_3376`
            ELSE CAST(NULL AS FLOAT64)
          END
          ELSE CAST(NULL AS FLOAT64)
        END AS `bfuid_col_3393`
      FROM (
        SELECT
          *
        FROM (
          SELECT
            *
          FROM UNNEST(ARRAY<STRUCT<`col_0` STRING, `col_1` INT64, `col_2` INT64>>[STRUCT('TEMP', 0, 0)]) AS `col_0`
        ) AS `t0`
      ) AS `t10`
      CROSS JOIN (
        SELECT
          `t19`.`bfuid_col_3376`,
          0 AS `bfuid_col_3389`
        FROM (
          SELECT
            COALESCE(SUM(`t17`.`bfuid_col_3376`), 0) AS `bfuid_col_3376`
          FROM (
            SELECT
              (
                `t15`.`bfuid_col_3330` * (
                  1 - `t15`.`bfuid_col_3331`
                )
              ) * CAST(regexp_contains(`t15`.`bfuid_col_3320`, 'PROMO') AS INT64) AS `bfuid_col_3376`
            FROM (
              SELECT
                `t12`.`P_TYPE` AS `bfuid_col_3320`,
                `t13`.`L_EXTENDEDPRICE` AS `bfuid_col_3330`,
                `t13`.`L_DISCOUNT` AS `bfuid_col_3331`,
                (
                  `t13`.`L_SHIPDATE` >= DATE(1995, 9, 1)
                )
                AND (
                  `t13`.`L_SHIPDATE` < DATE(1995, 10, 1)
                ) AS `bfuid_col_3341`
              FROM (
                SELECT
                  `t2`.`P_PARTKEY`,
                  `t2`.`P_TYPE`
                FROM (
                  SELECT
                    `P_PARTKEY`,
                    `P_TYPE`
                  FROM `bigframes-dev.tpch.PART` FOR SYSTEM_TIME AS OF CAST('2026-03-10T18:00:00' AS DATETIME)
                ) AS `t2`
              ) AS `t12`
              INNER JOIN (
                SELECT
                  `t3`.`L_PARTKEY`,
                  `t3`.`L_EXTENDEDPRICE`,
                  `t3`.`L_DISCOUNT`,
                  `t3`.`L_SHIPDATE`
                FROM (
                  SELECT
                    `L_PARTKEY`,
                    `L_EXTENDEDPRICE`,
                    `L_DISCOUNT`,
                    `L_SHIPDATE`
                  FROM `bigframes-dev.tpch.LINEITEM` FOR SYSTEM_TIME AS OF CAST('2026-03-10T18:00:00' AS DATETIME)
                ) AS `t3`
              ) AS `t13`
                ON COALESCE(`t12`.`P_PARTKEY`, 0) = COALESCE(`t13`.`L_PARTKEY`, 0)
                AND COALESCE(`t12`.`P_PARTKEY`, 1) = COALESCE(`t13`.`L_PARTKEY`, 1)
            ) AS `t15`
            WHERE
              `t15`.`bfuid_col_3341`
          ) AS `t17`
        ) AS `t19`
      ) AS `t22`
    ) AS `t25`
    GROUP BY
      1,
      2
  ) AS `t27`
  WHERE
    (
      `t27`.`bfuid_col_3391`
    ) IS NOT NULL
    AND (
      `t27`.`bfuid_col_3392`
    ) IS NOT NULL
) AS `t30`
FULL OUTER JOIN (
  SELECT
    `t28`.`bfuid_col_3384`,
    `t28`.`bfuid_col_3387`
  FROM (
    SELECT
      `t26`.`bfuid_col_3384`,
      `t26`.`bfuid_col_3385`,
      ANY_VALUE(`t26`.`bfuid_col_3386`) AS `bfuid_col_3387`
    FROM (
      SELECT
        `t11`.`col_0` AS `bfuid_col_3384`,
        `t11`.`col_1` AS `bfuid_col_3385`,
        CASE
          WHEN `t24`.`bfuid_col_3382` = 0
          THEN CASE
            WHEN `t11`.`col_2` = 0
            THEN `t24`.`bfuid_col_3380`
            ELSE CAST(NULL AS FLOAT64)
          END
          ELSE CAST(NULL AS FLOAT64)
        END AS `bfuid_col_3386`
      FROM (
        SELECT
          *
        FROM (
          SELECT
            *
          FROM UNNEST(ARRAY<STRUCT<`col_0` STRING, `col_1` INT64, `col_2` INT64>>[STRUCT('TEMP', 0, 0)]) AS `col_0`
        ) AS `t1`
      ) AS `t11`
      CROSS JOIN (
        SELECT
          `t20`.`bfuid_col_3380`,
          0 AS `bfuid_col_3382`
        FROM (
          SELECT
            COALESCE(SUM(`t18`.`bfuid_col_3380`), 0) AS `bfuid_col_3380`
          FROM (
            SELECT
              `t16`.`bfuid_col_3330` * (
                1 - `t16`.`bfuid_col_3331`
              ) AS `bfuid_col_3380`
            FROM (
              SELECT
                `t13`.`L_EXTENDEDPRICE` AS `bfuid_col_3330`,
                `t13`.`L_DISCOUNT` AS `bfuid_col_3331`,
                (
                  `t13`.`L_SHIPDATE` >= DATE(1995, 9, 1)
                )
                AND (
                  `t13`.`L_SHIPDATE` < DATE(1995, 10, 1)
                ) AS `bfuid_col_3341`
              FROM (
                SELECT
                  `t4`.`P_PARTKEY`
                FROM (
                  SELECT
                    `P_PARTKEY`
                  FROM `bigframes-dev.tpch.PART` FOR SYSTEM_TIME AS OF CAST('2026-03-10T18:00:00' AS DATETIME)
                ) AS `t4`
              ) AS `t14`
              INNER JOIN (
                SELECT
                  `t3`.`L_PARTKEY`,
                  `t3`.`L_EXTENDEDPRICE`,
                  `t3`.`L_DISCOUNT`,
                  `t3`.`L_SHIPDATE`
                FROM (
                  SELECT
                    `L_PARTKEY`,
                    `L_EXTENDEDPRICE`,
                    `L_DISCOUNT`,
                    `L_SHIPDATE`
                  FROM `bigframes-dev.tpch.LINEITEM` FOR SYSTEM_TIME AS OF CAST('2026-03-10T18:00:00' AS DATETIME)
                ) AS `t3`
              ) AS `t13`
                ON COALESCE(`t14`.`P_PARTKEY`, 0) = COALESCE(`t13`.`L_PARTKEY`, 0)
                AND COALESCE(`t14`.`P_PARTKEY`, 1) = COALESCE(`t13`.`L_PARTKEY`, 1)
            ) AS `t16`
            WHERE
              `t16`.`bfuid_col_3341`
          ) AS `t18`
        ) AS `t20`
      ) AS `t24`
    ) AS `t26`
    GROUP BY
      1,
      2
  ) AS `t28`
  WHERE
    (
      `t28`.`bfuid_col_3384`
    ) IS NOT NULL
    AND (
      `t28`.`bfuid_col_3385`
    ) IS NOT NULL
) AS `t32`
  ON `t30`.`bfuid_col_3391` = `t32`.`bfuid_col_3384`) AS `t`
ORDER BY `bfuid_col_3403` ASC NULLS LAST