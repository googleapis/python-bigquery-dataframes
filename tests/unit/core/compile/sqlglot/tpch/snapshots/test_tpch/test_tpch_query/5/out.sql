SELECT `N_NAME`, `REVENUE` FROM (SELECT
  `t32`.`N_NAME`,
  `t32`.`bfuid_col_2182` AS `REVENUE`
FROM (
  SELECT
    `t31`.`N_NAME`,
    COALESCE(SUM(`t31`.`bfuid_col_2181`), 0) AS `bfuid_col_2182`
  FROM (
    SELECT
      `t30`.`bfuid_col_2181`,
      `t30`.`N_NAME`
    FROM (
      SELECT
        *
      FROM (
        SELECT
          `t16`.`bfuid_col_2167`,
          `t16`.`bfuid_col_2181`,
          `t27`.`N_NATIONKEY`,
          `t27`.`N_NAME`
        FROM (
          SELECT
            `t1`.`L_ORDERKEY` AS `bfuid_col_2165`,
            `t1`.`L_SUPPKEY` AS `bfuid_col_2167`,
            `t1`.`L_EXTENDEDPRICE` * (
              1.0 - `t1`.`L_DISCOUNT`
            ) AS `bfuid_col_2181`
          FROM (
            SELECT
              `L_ORDERKEY`,
              `L_SUPPKEY`,
              `L_EXTENDEDPRICE`,
              `L_DISCOUNT`
            FROM `bigframes-dev.tpch.LINEITEM` FOR SYSTEM_TIME AS OF CAST('2026-03-10T18:00:00' AS DATETIME)
          ) AS `t1`
        ) AS `t16`
        INNER JOIN (
          SELECT
            *
          FROM (
            SELECT
              `t18`.`bfuid_col_2151`,
              `t24`.`N_NATIONKEY`,
              `t24`.`N_NAME`
            FROM (
              SELECT
                `t14`.`bfuid_col_2151`,
                `t14`.`bfuid_col_2152`
              FROM (
                SELECT
                  `t2`.`O_ORDERKEY` AS `bfuid_col_2151`,
                  `t2`.`O_CUSTKEY` AS `bfuid_col_2152`,
                  (
                    `t2`.`O_ORDERDATE` >= DATE(1994, 1, 1)
                  )
                  AND (
                    `t2`.`O_ORDERDATE` < DATE(1995, 1, 1)
                  ) AS `bfuid_col_2160`
                FROM (
                  SELECT
                    `O_ORDERKEY`,
                    `O_CUSTKEY`,
                    `O_ORDERDATE`
                  FROM `bigframes-dev.tpch.ORDERS` FOR SYSTEM_TIME AS OF CAST('2026-03-10T18:00:00' AS DATETIME)
                ) AS `t2`
              ) AS `t14`
              WHERE
                `t14`.`bfuid_col_2160`
            ) AS `t18`
            INNER JOIN (
              SELECT
                *
              FROM (
                SELECT
                  `t21`.`N_NATIONKEY`,
                  `t21`.`N_NAME`,
                  `t10`.`C_CUSTKEY`
                FROM (
                  SELECT
                    *
                  FROM (
                    SELECT
                      `t12`.`N_NATIONKEY`,
                      `t12`.`N_NAME`
                    FROM (
                      SELECT
                        `t4`.`R_REGIONKEY` AS `bfuid_col_2142`
                      FROM (
                        SELECT
                          `R_REGIONKEY`,
                          `R_NAME`
                        FROM `bigframes-dev.tpch.REGION` FOR SYSTEM_TIME AS OF CAST('2026-03-10T18:00:00' AS DATETIME)
                      ) AS `t4`
                      WHERE
                        `t4`.`R_NAME` = 'ASIA'
                    ) AS `t15`
                    INNER JOIN (
                      SELECT
                        `t5`.`N_NATIONKEY`,
                        `t5`.`N_NAME`,
                        `t5`.`N_REGIONKEY`
                      FROM (
                        SELECT
                          `N_NATIONKEY`,
                          `N_NAME`,
                          `N_REGIONKEY`
                        FROM `bigframes-dev.tpch.NATION` FOR SYSTEM_TIME AS OF CAST('2026-03-10T18:00:00' AS DATETIME)
                      ) AS `t5`
                    ) AS `t12`
                      ON COALESCE(`t15`.`bfuid_col_2142`, 0) = COALESCE(`t12`.`N_REGIONKEY`, 0)
                      AND COALESCE(`t15`.`bfuid_col_2142`, 1) = COALESCE(`t12`.`N_REGIONKEY`, 1)
                  ) AS `t19`
                ) AS `t21`
                INNER JOIN (
                  SELECT
                    `t3`.`C_CUSTKEY`,
                    `t3`.`C_NATIONKEY`
                  FROM (
                    SELECT
                      `C_CUSTKEY`,
                      `C_NATIONKEY`
                    FROM `bigframes-dev.tpch.CUSTOMER` FOR SYSTEM_TIME AS OF CAST('2026-03-10T18:00:00' AS DATETIME)
                  ) AS `t3`
                ) AS `t10`
                  ON COALESCE(`t21`.`N_NATIONKEY`, 0) = COALESCE(`t10`.`C_NATIONKEY`, 0)
                  AND COALESCE(`t21`.`N_NATIONKEY`, 1) = COALESCE(`t10`.`C_NATIONKEY`, 1)
              ) AS `t22`
            ) AS `t24`
              ON COALESCE(`t18`.`bfuid_col_2152`, 0) = COALESCE(`t24`.`C_CUSTKEY`, 0)
              AND COALESCE(`t18`.`bfuid_col_2152`, 1) = COALESCE(`t24`.`C_CUSTKEY`, 1)
          ) AS `t25`
        ) AS `t27`
          ON COALESCE(`t16`.`bfuid_col_2165`, 0) = COALESCE(`t27`.`bfuid_col_2151`, 0)
          AND COALESCE(`t16`.`bfuid_col_2165`, 1) = COALESCE(`t27`.`bfuid_col_2151`, 1)
      ) AS `t28`
    ) AS `t30`
    INNER JOIN (
      SELECT
        `t0`.`S_SUPPKEY`,
        `t0`.`S_NATIONKEY`
      FROM (
        SELECT
          `S_SUPPKEY`,
          `S_NATIONKEY`
        FROM `bigframes-dev.tpch.SUPPLIER` FOR SYSTEM_TIME AS OF CAST('2026-03-10T18:00:00' AS DATETIME)
      ) AS `t0`
    ) AS `t9`
      ON COALESCE(`t30`.`bfuid_col_2167`, 0) = COALESCE(`t9`.`S_SUPPKEY`, 0)
      AND COALESCE(`t30`.`bfuid_col_2167`, 1) = COALESCE(`t9`.`S_SUPPKEY`, 1)
      AND COALESCE(`t30`.`N_NATIONKEY`, 0) = COALESCE(`t9`.`S_NATIONKEY`, 0)
      AND COALESCE(`t30`.`N_NATIONKEY`, 1) = COALESCE(`t9`.`S_NATIONKEY`, 1)
  ) AS `t31`
  GROUP BY
    1
) AS `t32`
WHERE
  (
    `t32`.`N_NAME`
  ) IS NOT NULL) AS `t`
ORDER BY `REVENUE` DESC NULLS LAST ,`N_NAME` ASC NULLS LAST