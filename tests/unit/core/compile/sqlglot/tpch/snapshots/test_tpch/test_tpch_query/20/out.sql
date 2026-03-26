SELECT `S_NAME`, `S_ADDRESS` FROM (SELECT
  `t26`.`S_NAME`,
  `t26`.`S_ADDRESS`
FROM (
  SELECT
    `t14`.`S_NAME`,
    `t14`.`S_ADDRESS`,
    EXISTS(
      SELECT
        1
      FROM (
        SELECT
          `t23`.`bfuid_col_3812`
        FROM (
          SELECT
            `t22`.`bfuid_col_3812`
          FROM (
            SELECT
              `t21`.`PS_SUPPKEY` AS `bfuid_col_3812`,
              `t21`.`PS_AVAILQTY` > `t19`.`bfuid_col_3778` AS `bfuid_col_3816`
            FROM (
              SELECT
                `t15`.`bfuid_col_3758` AS `bfuid_col_3776`,
                `t15`.`bfuid_col_3759` AS `bfuid_col_3777`,
                `t15`.`bfuid_col_3774` * 0.5 AS `bfuid_col_3778`
              FROM (
                SELECT
                  `t13`.`bfuid_col_3758`,
                  `t13`.`bfuid_col_3759`,
                  COALESCE(SUM(`t13`.`bfuid_col_3761`), 0) AS `bfuid_col_3774`
                FROM (
                  SELECT
                    `t11`.`bfuid_col_3758`,
                    `t11`.`bfuid_col_3759`,
                    `t11`.`bfuid_col_3761`
                  FROM (
                    SELECT
                      `t2`.`L_PARTKEY` AS `bfuid_col_3758`,
                      `t2`.`L_SUPPKEY` AS `bfuid_col_3759`,
                      `t2`.`L_QUANTITY` AS `bfuid_col_3761`,
                      (
                        `t2`.`L_SHIPDATE` >= DATE(1994, 1, 1)
                      )
                      AND (
                        `t2`.`L_SHIPDATE` < DATE(1995, 1, 1)
                      ) AS `bfuid_col_3773`
                    FROM (
                      SELECT
                        `L_PARTKEY`,
                        `L_SUPPKEY`,
                        `L_QUANTITY`,
                        `L_SHIPDATE`
                      FROM `bigframes-dev.tpch.LINEITEM` FOR SYSTEM_TIME AS OF CAST('2026-03-10T18:00:00' AS DATETIME)
                    ) AS `t2`
                  ) AS `t11`
                  WHERE
                    `t11`.`bfuid_col_3773`
                ) AS `t13`
                GROUP BY
                  1,
                  2
              ) AS `t15`
              WHERE
                (
                  `t15`.`bfuid_col_3758`
                ) IS NOT NULL
                AND (
                  `t15`.`bfuid_col_3759`
                ) IS NOT NULL
            ) AS `t19`
            INNER JOIN (
              SELECT
                `t17`.`PS_PARTKEY`,
                `t17`.`PS_SUPPKEY`,
                `t17`.`PS_AVAILQTY`
              FROM (
                SELECT
                  `t6`.`PS_PARTKEY`,
                  `t6`.`PS_SUPPKEY`,
                  `t6`.`PS_AVAILQTY`,
                  EXISTS(
                    SELECT
                      1
                    FROM (
                      SELECT
                        `t9`.`bfuid_col_3787`
                      FROM (
                        SELECT
                          `t4`.`P_PARTKEY` AS `bfuid_col_3787`
                        FROM (
                          SELECT
                            `P_PARTKEY`,
                            `P_NAME`
                          FROM `bigframes-dev.tpch.PART` FOR SYSTEM_TIME AS OF CAST('2026-03-10T18:00:00' AS DATETIME)
                        ) AS `t4`
                        WHERE
                          STARTS_WITH(`t4`.`P_NAME`, 'forest')
                      ) AS `t9`
                      GROUP BY
                        1
                    ) AS `t12`
                    WHERE
                      (
                        COALESCE(`t6`.`PS_PARTKEY`, 0) = COALESCE(`t12`.`bfuid_col_3787`, 0)
                      )
                      AND (
                        COALESCE(`t6`.`PS_PARTKEY`, 1) = COALESCE(`t12`.`bfuid_col_3787`, 1)
                      )
                  ) AS `bfuid_col_3797`
                FROM (
                  SELECT
                    `t3`.`PS_PARTKEY`,
                    `t3`.`PS_SUPPKEY`,
                    `t3`.`PS_AVAILQTY`
                  FROM (
                    SELECT
                      `PS_PARTKEY`,
                      `PS_SUPPKEY`,
                      `PS_AVAILQTY`
                    FROM `bigframes-dev.tpch.PARTSUPP` FOR SYSTEM_TIME AS OF CAST('2026-03-10T18:00:00' AS DATETIME)
                  ) AS `t3`
                ) AS `t6`
              ) AS `t17`
              WHERE
                `t17`.`bfuid_col_3797`
            ) AS `t21`
              ON `t19`.`bfuid_col_3777` = `t21`.`PS_SUPPKEY`
              AND `t19`.`bfuid_col_3776` = `t21`.`PS_PARTKEY`
          ) AS `t22`
          WHERE
            `t22`.`bfuid_col_3816`
        ) AS `t23`
        GROUP BY
          1
      ) AS `t24`
      WHERE
        (
          COALESCE(`t14`.`S_SUPPKEY`, 0) = COALESCE(`t24`.`bfuid_col_3812`, 0)
        )
        AND (
          COALESCE(`t14`.`S_SUPPKEY`, 1) = COALESCE(`t24`.`bfuid_col_3812`, 1)
        )
    ) AS `bfuid_col_3817`
  FROM (
    SELECT
      `t7`.`S_SUPPKEY`,
      `t7`.`S_NAME`,
      `t7`.`S_ADDRESS`
    FROM (
      SELECT
        `t0`.`S_SUPPKEY`,
        `t0`.`S_NAME`,
        `t0`.`S_ADDRESS`,
        `t0`.`S_NATIONKEY`
      FROM (
        SELECT
          `S_SUPPKEY`,
          `S_NAME`,
          `S_ADDRESS`,
          `S_NATIONKEY`
        FROM `bigframes-dev.tpch.SUPPLIER` FOR SYSTEM_TIME AS OF CAST('2026-03-10T18:00:00' AS DATETIME)
      ) AS `t0`
    ) AS `t7`
    INNER JOIN (
      SELECT
        `t1`.`N_NATIONKEY` AS `bfuid_col_3781`
      FROM (
        SELECT
          `N_NATIONKEY`,
          `N_NAME`
        FROM `bigframes-dev.tpch.NATION` FOR SYSTEM_TIME AS OF CAST('2026-03-10T18:00:00' AS DATETIME)
      ) AS `t1`
      WHERE
        `t1`.`N_NAME` = 'CANADA'
    ) AS `t10`
      ON COALESCE(`t7`.`S_NATIONKEY`, 0) = COALESCE(`t10`.`bfuid_col_3781`, 0)
      AND COALESCE(`t7`.`S_NATIONKEY`, 1) = COALESCE(`t10`.`bfuid_col_3781`, 1)
  ) AS `t14`
) AS `t26`
WHERE
  `t26`.`bfuid_col_3817`) AS `t`
ORDER BY `S_NAME` ASC NULLS LAST