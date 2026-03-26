SELECT `NATION`, `O_YEAR`, `SUM_PROFIT` FROM (SELECT
  `t32`.`bfuid_col_3032` AS `NATION`,
  `t32`.`bfuid_col_3035` AS `O_YEAR`,
  ROUND(`t32`.`bfuid_col_3037`, 2) AS `SUM_PROFIT`,
  `t32`.`bfuid_col_3035` AS `bfuid_col_3045`,
  `t32`.`bfuid_col_3032` AS `bfuid_col_3046`
FROM (
  SELECT
    `t31`.`bfuid_col_3032`,
    `t31`.`bfuid_col_3035`,
    COALESCE(SUM(`t31`.`bfuid_col_3036`), 0) AS `bfuid_col_3037`
  FROM (
    SELECT
      `t30`.`bfuid_col_2919` AS `bfuid_col_3032`,
      EXTRACT(year FROM `t30`.`bfuid_col_2913`) AS `bfuid_col_3035`,
      (
        `t30`.`bfuid_col_2886` * (
          1 - `t30`.`bfuid_col_2887`
        )
      ) - (
        `t30`.`bfuid_col_2900` * `t30`.`bfuid_col_2885`
      ) AS `bfuid_col_3036`
    FROM (
      SELECT
        `t29`.`L_QUANTITY` AS `bfuid_col_2885`,
        `t29`.`L_EXTENDEDPRICE` AS `bfuid_col_2886`,
        `t29`.`L_DISCOUNT` AS `bfuid_col_2887`,
        `t29`.`PS_SUPPLYCOST` AS `bfuid_col_2900`,
        `t29`.`O_ORDERDATE` AS `bfuid_col_2913`,
        `t12`.`N_NAME` AS `bfuid_col_2919`,
        regexp_contains(`t29`.`P_NAME`, 'green') AS `bfuid_col_2922`
      FROM (
        SELECT
          *
        FROM (
          SELECT
            `t26`.`P_NAME`,
            `t26`.`L_QUANTITY`,
            `t26`.`L_EXTENDEDPRICE`,
            `t26`.`L_DISCOUNT`,
            `t26`.`PS_SUPPLYCOST`,
            `t26`.`S_NATIONKEY`,
            `t13`.`O_ORDERDATE`
          FROM (
            SELECT
              *
            FROM (
              SELECT
                `t23`.`P_NAME`,
                `t23`.`L_ORDERKEY`,
                `t23`.`L_QUANTITY`,
                `t23`.`L_EXTENDEDPRICE`,
                `t23`.`L_DISCOUNT`,
                `t23`.`PS_SUPPLYCOST`,
                `t14`.`S_NATIONKEY`
              FROM (
                SELECT
                  *
                FROM (
                  SELECT
                    `t20`.`P_NAME`,
                    `t20`.`L_ORDERKEY`,
                    `t20`.`L_SUPPKEY`,
                    `t20`.`L_QUANTITY`,
                    `t20`.`L_EXTENDEDPRICE`,
                    `t20`.`L_DISCOUNT`,
                    `t15`.`PS_SUPPLYCOST`
                  FROM (
                    SELECT
                      *
                    FROM (
                      SELECT
                        `t16`.`P_NAME`,
                        `t17`.`L_ORDERKEY`,
                        `t17`.`L_PARTKEY`,
                        `t17`.`L_SUPPKEY`,
                        `t17`.`L_QUANTITY`,
                        `t17`.`L_EXTENDEDPRICE`,
                        `t17`.`L_DISCOUNT`
                      FROM (
                        SELECT
                          `t4`.`P_PARTKEY`,
                          `t4`.`P_NAME`
                        FROM (
                          SELECT
                            `P_PARTKEY`,
                            `P_NAME`
                          FROM `bigframes-dev.tpch.PART` FOR SYSTEM_TIME AS OF CAST('2026-03-10T18:00:00' AS DATETIME)
                        ) AS `t4`
                      ) AS `t16`
                      INNER JOIN (
                        SELECT
                          `t5`.`L_ORDERKEY`,
                          `t5`.`L_PARTKEY`,
                          `t5`.`L_SUPPKEY`,
                          `t5`.`L_QUANTITY`,
                          `t5`.`L_EXTENDEDPRICE`,
                          `t5`.`L_DISCOUNT`
                        FROM (
                          SELECT
                            `L_ORDERKEY`,
                            `L_PARTKEY`,
                            `L_SUPPKEY`,
                            `L_QUANTITY`,
                            `L_EXTENDEDPRICE`,
                            `L_DISCOUNT`
                          FROM `bigframes-dev.tpch.LINEITEM` FOR SYSTEM_TIME AS OF CAST('2026-03-10T18:00:00' AS DATETIME)
                        ) AS `t5`
                      ) AS `t17`
                        ON COALESCE(`t16`.`P_PARTKEY`, 0) = COALESCE(`t17`.`L_PARTKEY`, 0)
                        AND COALESCE(`t16`.`P_PARTKEY`, 1) = COALESCE(`t17`.`L_PARTKEY`, 1)
                    ) AS `t18`
                  ) AS `t20`
                  INNER JOIN (
                    SELECT
                      `t3`.`PS_PARTKEY`,
                      `t3`.`PS_SUPPKEY`,
                      `t3`.`PS_SUPPLYCOST`
                    FROM (
                      SELECT
                        `PS_PARTKEY`,
                        `PS_SUPPKEY`,
                        `PS_SUPPLYCOST`
                      FROM `bigframes-dev.tpch.PARTSUPP` FOR SYSTEM_TIME AS OF CAST('2026-03-10T18:00:00' AS DATETIME)
                    ) AS `t3`
                  ) AS `t15`
                    ON COALESCE(`t20`.`L_SUPPKEY`, 0) = COALESCE(`t15`.`PS_SUPPKEY`, 0)
                    AND COALESCE(`t20`.`L_SUPPKEY`, 1) = COALESCE(`t15`.`PS_SUPPKEY`, 1)
                    AND COALESCE(`t20`.`L_PARTKEY`, 0) = COALESCE(`t15`.`PS_PARTKEY`, 0)
                    AND COALESCE(`t20`.`L_PARTKEY`, 1) = COALESCE(`t15`.`PS_PARTKEY`, 1)
                ) AS `t21`
              ) AS `t23`
              INNER JOIN (
                SELECT
                  `t2`.`S_SUPPKEY`,
                  `t2`.`S_NATIONKEY`
                FROM (
                  SELECT
                    `S_SUPPKEY`,
                    `S_NATIONKEY`
                  FROM `bigframes-dev.tpch.SUPPLIER` FOR SYSTEM_TIME AS OF CAST('2026-03-10T18:00:00' AS DATETIME)
                ) AS `t2`
              ) AS `t14`
                ON COALESCE(`t23`.`L_SUPPKEY`, 0) = COALESCE(`t14`.`S_SUPPKEY`, 0)
                AND COALESCE(`t23`.`L_SUPPKEY`, 1) = COALESCE(`t14`.`S_SUPPKEY`, 1)
            ) AS `t24`
          ) AS `t26`
          INNER JOIN (
            SELECT
              `t1`.`O_ORDERKEY`,
              `t1`.`O_ORDERDATE`
            FROM (
              SELECT
                `O_ORDERKEY`,
                `O_ORDERDATE`
              FROM `bigframes-dev.tpch.ORDERS` FOR SYSTEM_TIME AS OF CAST('2026-03-10T18:00:00' AS DATETIME)
            ) AS `t1`
          ) AS `t13`
            ON COALESCE(`t26`.`L_ORDERKEY`, 0) = COALESCE(`t13`.`O_ORDERKEY`, 0)
            AND COALESCE(`t26`.`L_ORDERKEY`, 1) = COALESCE(`t13`.`O_ORDERKEY`, 1)
        ) AS `t27`
      ) AS `t29`
      INNER JOIN (
        SELECT
          `t0`.`N_NATIONKEY`,
          `t0`.`N_NAME`
        FROM (
          SELECT
            `N_NATIONKEY`,
            `N_NAME`
          FROM `bigframes-dev.tpch.NATION` FOR SYSTEM_TIME AS OF CAST('2026-03-10T18:00:00' AS DATETIME)
        ) AS `t0`
      ) AS `t12`
        ON COALESCE(`t29`.`S_NATIONKEY`, 0) = COALESCE(`t12`.`N_NATIONKEY`, 0)
        AND COALESCE(`t29`.`S_NATIONKEY`, 1) = COALESCE(`t12`.`N_NATIONKEY`, 1)
    ) AS `t30`
    WHERE
      `t30`.`bfuid_col_2922`
  ) AS `t31`
  GROUP BY
    1,
    2
) AS `t32`
WHERE
  (
    `t32`.`bfuid_col_3032`
  ) IS NOT NULL
  AND (
    `t32`.`bfuid_col_3035`
  ) IS NOT NULL) AS `t`
ORDER BY `NATION` ASC NULLS LAST ,`O_YEAR` DESC NULLS LAST ,`bfuid_col_3046` ASC NULLS LAST ,`bfuid_col_3045` ASC NULLS LAST