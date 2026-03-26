SELECT `S_SUPPKEY`, `S_NAME`, `S_ADDRESS`, `S_PHONE`, `TOTAL_REVENUE` FROM (SELECT
  `t26`.`bfuid_col_3466` AS `S_SUPPKEY`,
  `t26`.`bfuid_col_3467` AS `S_NAME`,
  `t26`.`bfuid_col_3468` AS `S_ADDRESS`,
  `t26`.`bfuid_col_3470` AS `S_PHONE`,
  `t26`.`bfuid_col_3474` AS `TOTAL_REVENUE`
FROM (
  SELECT
    `t18`.`S_SUPPKEY` AS `bfuid_col_3466`,
    `t18`.`S_NAME` AS `bfuid_col_3467`,
    `t18`.`S_ADDRESS` AS `bfuid_col_3468`,
    `t18`.`S_PHONE` AS `bfuid_col_3470`,
    `t18`.`bfuid_col_3450` AS `bfuid_col_3474`,
    `t18`.`bfuid_col_3450` = `t25`.`bfuid_col_3458` AS `bfuid_col_3476`
  FROM (
    SELECT
      *
    FROM (
      SELECT
        `t7`.`S_SUPPKEY`,
        `t7`.`S_NAME`,
        `t7`.`S_ADDRESS`,
        `t7`.`S_PHONE`,
        `t14`.`bfuid_col_3450`
      FROM (
        SELECT
          `t0`.`S_SUPPKEY`,
          `t0`.`S_NAME`,
          `t0`.`S_ADDRESS`,
          `t0`.`S_PHONE`
        FROM (
          SELECT
            `S_SUPPKEY`,
            `S_NAME`,
            `S_ADDRESS`,
            `S_PHONE`
          FROM `bigframes-dev.tpch.SUPPLIER` FOR SYSTEM_TIME AS OF CAST('2026-03-10T18:00:00' AS DATETIME)
        ) AS `t0`
      ) AS `t7`
      INNER JOIN (
        SELECT
          `t12`.`bfuid_col_3432` AS `bfuid_col_3449`,
          ROUND(`t12`.`bfuid_col_3447`, 2) AS `bfuid_col_3450`
        FROM (
          SELECT
            `t11`.`bfuid_col_3432`,
            COALESCE(SUM(`t11`.`bfuid_col_3446`), 0) AS `bfuid_col_3447`
          FROM (
            SELECT
              `t10`.`bfuid_col_3411` AS `bfuid_col_3432`,
              `t10`.`bfuid_col_3414` * (
                1 - `t10`.`bfuid_col_3415`
              ) AS `bfuid_col_3446`
            FROM (
              SELECT
                `t2`.`L_SUPPKEY` AS `bfuid_col_3411`,
                `t2`.`L_EXTENDEDPRICE` AS `bfuid_col_3414`,
                `t2`.`L_DISCOUNT` AS `bfuid_col_3415`,
                (
                  `t2`.`L_SHIPDATE` >= DATE(1996, 1, 1)
                )
                AND (
                  `t2`.`L_SHIPDATE` < DATE(1996, 4, 1)
                ) AS `bfuid_col_3425`
              FROM (
                SELECT
                  `L_SUPPKEY`,
                  `L_EXTENDEDPRICE`,
                  `L_DISCOUNT`,
                  `L_SHIPDATE`
                FROM `bigframes-dev.tpch.LINEITEM` FOR SYSTEM_TIME AS OF CAST('2026-03-10T18:00:00' AS DATETIME)
              ) AS `t2`
            ) AS `t10`
            WHERE
              `t10`.`bfuid_col_3425`
          ) AS `t11`
          GROUP BY
            1
        ) AS `t12`
        WHERE
          (
            `t12`.`bfuid_col_3432`
          ) IS NOT NULL
      ) AS `t14`
        ON `t7`.`S_SUPPKEY` = `t14`.`bfuid_col_3449`
    ) AS `t15`
  ) AS `t18`
  CROSS JOIN (
    SELECT
      `t23`.`bfuid_col_3458`
    FROM (
      SELECT
        `t22`.`bfuid_col_3455`,
        `t22`.`bfuid_col_3456`,
        ANY_VALUE(`t22`.`bfuid_col_3457`) AS `bfuid_col_3458`
      FROM (
        SELECT
          `t8`.`col_0` AS `bfuid_col_3455`,
          `t8`.`col_1` AS `bfuid_col_3456`,
          CASE
            WHEN `t21`.`bfuid_col_3453` = 0
            THEN CASE
              WHEN `t8`.`col_2` = 0
              THEN `t21`.`bfuid_col_3450`
              ELSE CAST(NULL AS FLOAT64)
            END
            ELSE CAST(NULL AS FLOAT64)
          END AS `bfuid_col_3457`
        FROM (
          SELECT
            *
          FROM (
            SELECT
              *
            FROM UNNEST(ARRAY<STRUCT<`col_0` STRING, `col_1` INT64, `col_2` INT64>>[STRUCT('TOTAL_REVENUE', 0, 0)]) AS `col_0`
          ) AS `t1`
        ) AS `t8`
        CROSS JOIN (
          SELECT
            `t19`.`bfuid_col_3450`,
            0 AS `bfuid_col_3453`
          FROM (
            SELECT
              MAX(`t16`.`bfuid_col_3450`) AS `bfuid_col_3450`
            FROM (
              SELECT
                `t14`.`bfuid_col_3450`
              FROM (
                SELECT
                  `t3`.`S_SUPPKEY`
                FROM (
                  SELECT
                    `S_SUPPKEY`
                  FROM `bigframes-dev.tpch.SUPPLIER` FOR SYSTEM_TIME AS OF CAST('2026-03-10T18:00:00' AS DATETIME)
                ) AS `t3`
              ) AS `t9`
              INNER JOIN (
                SELECT
                  `t12`.`bfuid_col_3432` AS `bfuid_col_3449`,
                  ROUND(`t12`.`bfuid_col_3447`, 2) AS `bfuid_col_3450`
                FROM (
                  SELECT
                    `t11`.`bfuid_col_3432`,
                    COALESCE(SUM(`t11`.`bfuid_col_3446`), 0) AS `bfuid_col_3447`
                  FROM (
                    SELECT
                      `t10`.`bfuid_col_3411` AS `bfuid_col_3432`,
                      `t10`.`bfuid_col_3414` * (
                        1 - `t10`.`bfuid_col_3415`
                      ) AS `bfuid_col_3446`
                    FROM (
                      SELECT
                        `t2`.`L_SUPPKEY` AS `bfuid_col_3411`,
                        `t2`.`L_EXTENDEDPRICE` AS `bfuid_col_3414`,
                        `t2`.`L_DISCOUNT` AS `bfuid_col_3415`,
                        (
                          `t2`.`L_SHIPDATE` >= DATE(1996, 1, 1)
                        )
                        AND (
                          `t2`.`L_SHIPDATE` < DATE(1996, 4, 1)
                        ) AS `bfuid_col_3425`
                      FROM (
                        SELECT
                          `L_SUPPKEY`,
                          `L_EXTENDEDPRICE`,
                          `L_DISCOUNT`,
                          `L_SHIPDATE`
                        FROM `bigframes-dev.tpch.LINEITEM` FOR SYSTEM_TIME AS OF CAST('2026-03-10T18:00:00' AS DATETIME)
                      ) AS `t2`
                    ) AS `t10`
                    WHERE
                      `t10`.`bfuid_col_3425`
                  ) AS `t11`
                  GROUP BY
                    1
                ) AS `t12`
                WHERE
                  (
                    `t12`.`bfuid_col_3432`
                  ) IS NOT NULL
              ) AS `t14`
                ON `t9`.`S_SUPPKEY` = `t14`.`bfuid_col_3449`
            ) AS `t16`
          ) AS `t19`
        ) AS `t21`
      ) AS `t22`
      GROUP BY
        1,
        2
    ) AS `t23`
    WHERE
      (
        `t23`.`bfuid_col_3455`
      ) IS NOT NULL
      AND (
        `t23`.`bfuid_col_3456`
      ) IS NOT NULL
  ) AS `t25`
) AS `t26`
WHERE
  `t26`.`bfuid_col_3476`) AS `t`
ORDER BY `S_SUPPKEY` ASC NULLS LAST