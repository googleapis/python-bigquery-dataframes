SELECT `O_YEAR`, `MKT_SHARE` FROM (SELECT
  `t44`.`bfuid_col_2854` AS `O_YEAR`,
  ROUND(ieee_divide(`t44`.`bfuid_col_2858`, `t44`.`bfuid_col_2859`), 2) AS `MKT_SHARE`,
  `t44`.`bfuid_col_2854` AS `bfuid_col_2870`
FROM (
  SELECT
    `t43`.`bfuid_col_2854`,
    COALESCE(SUM(`t43`.`bfuid_col_2856`), 0) AS `bfuid_col_2858`,
    COALESCE(SUM(`t43`.`bfuid_col_2857`), 0) AS `bfuid_col_2859`
  FROM (
    SELECT
      EXTRACT(year FROM `t42`.`bfuid_col_2536`) AS `bfuid_col_2854`,
      CASE
        WHEN `t42`.`bfuid_col_2555` = 'BRAZIL'
        THEN `t42`.`bfuid_col_2514` * (
          1.0 - `t42`.`bfuid_col_2515`
        )
        ELSE 0.0
      END AS `bfuid_col_2856`,
      `t42`.`bfuid_col_2514` * (
        1.0 - `t42`.`bfuid_col_2515`
      ) AS `bfuid_col_2857`
    FROM (
      SELECT
        `t41`.`bfuid_col_2444` AS `bfuid_col_2504`,
        `t41`.`bfuid_col_2454` AS `bfuid_col_2514`,
        `t41`.`bfuid_col_2455` AS `bfuid_col_2515`,
        `t41`.`bfuid_col_2476` AS `bfuid_col_2536`,
        `t16`.`N_NAME` AS `bfuid_col_2555`,
        (
          `t41`.`bfuid_col_2476` >= DATE(1995, 1, 1)
        )
        AND (
          `t41`.`bfuid_col_2476` <= DATE(1996, 12, 31)
        ) AS `bfuid_col_2556`
      FROM (
        SELECT
          `t39`.`bfuid_col_2444`,
          `t39`.`bfuid_col_2454`,
          `t39`.`bfuid_col_2455`,
          `t39`.`bfuid_col_2468`,
          `t39`.`bfuid_col_2476`
        FROM (
          SELECT
            `t38`.`P_TYPE` AS `bfuid_col_2444`,
            `t38`.`L_EXTENDEDPRICE` AS `bfuid_col_2454`,
            `t38`.`L_DISCOUNT` AS `bfuid_col_2455`,
            `t38`.`S_NATIONKEY` AS `bfuid_col_2468`,
            `t38`.`O_ORDERDATE` AS `bfuid_col_2476`,
            `t17`.`R_NAME` = 'AMERICA' AS `bfuid_col_2494`
          FROM (
            SELECT
              *
            FROM (
              SELECT
                `t35`.`P_TYPE`,
                `t35`.`L_EXTENDEDPRICE`,
                `t35`.`L_DISCOUNT`,
                `t35`.`S_NATIONKEY`,
                `t35`.`O_ORDERDATE`,
                `t18`.`N_REGIONKEY`
              FROM (
                SELECT
                  *
                FROM (
                  SELECT
                    `t32`.`P_TYPE`,
                    `t32`.`L_EXTENDEDPRICE`,
                    `t32`.`L_DISCOUNT`,
                    `t32`.`S_NATIONKEY`,
                    `t32`.`O_ORDERDATE`,
                    `t19`.`C_NATIONKEY`
                  FROM (
                    SELECT
                      *
                    FROM (
                      SELECT
                        `t29`.`P_TYPE`,
                        `t29`.`L_EXTENDEDPRICE`,
                        `t29`.`L_DISCOUNT`,
                        `t29`.`S_NATIONKEY`,
                        `t20`.`O_CUSTKEY`,
                        `t20`.`O_ORDERDATE`
                      FROM (
                        SELECT
                          *
                        FROM (
                          SELECT
                            `t26`.`P_TYPE`,
                            `t26`.`L_ORDERKEY`,
                            `t26`.`L_EXTENDEDPRICE`,
                            `t26`.`L_DISCOUNT`,
                            `t21`.`S_NATIONKEY`
                          FROM (
                            SELECT
                              *
                            FROM (
                              SELECT
                                `t22`.`P_TYPE`,
                                `t23`.`L_ORDERKEY`,
                                `t23`.`L_SUPPKEY`,
                                `t23`.`L_EXTENDEDPRICE`,
                                `t23`.`L_DISCOUNT`
                              FROM (
                                SELECT
                                  `t6`.`P_PARTKEY`,
                                  `t6`.`P_TYPE`
                                FROM (
                                  SELECT
                                    `P_PARTKEY`,
                                    `P_TYPE`
                                  FROM `bigframes-dev.tpch.PART` FOR SYSTEM_TIME AS OF CAST('2026-03-10T18:00:00' AS DATETIME)
                                ) AS `t6`
                              ) AS `t22`
                              INNER JOIN (
                                SELECT
                                  `t7`.`L_ORDERKEY`,
                                  `t7`.`L_PARTKEY`,
                                  `t7`.`L_SUPPKEY`,
                                  `t7`.`L_EXTENDEDPRICE`,
                                  `t7`.`L_DISCOUNT`
                                FROM (
                                  SELECT
                                    `L_ORDERKEY`,
                                    `L_PARTKEY`,
                                    `L_SUPPKEY`,
                                    `L_EXTENDEDPRICE`,
                                    `L_DISCOUNT`
                                  FROM `bigframes-dev.tpch.LINEITEM` FOR SYSTEM_TIME AS OF CAST('2026-03-10T18:00:00' AS DATETIME)
                                ) AS `t7`
                              ) AS `t23`
                                ON COALESCE(`t22`.`P_PARTKEY`, 0) = COALESCE(`t23`.`L_PARTKEY`, 0)
                                AND COALESCE(`t22`.`P_PARTKEY`, 1) = COALESCE(`t23`.`L_PARTKEY`, 1)
                            ) AS `t24`
                          ) AS `t26`
                          INNER JOIN (
                            SELECT
                              `t5`.`S_SUPPKEY`,
                              `t5`.`S_NATIONKEY`
                            FROM (
                              SELECT
                                `S_SUPPKEY`,
                                `S_NATIONKEY`
                              FROM `bigframes-dev.tpch.SUPPLIER` FOR SYSTEM_TIME AS OF CAST('2026-03-10T18:00:00' AS DATETIME)
                            ) AS `t5`
                          ) AS `t21`
                            ON COALESCE(`t26`.`L_SUPPKEY`, 0) = COALESCE(`t21`.`S_SUPPKEY`, 0)
                            AND COALESCE(`t26`.`L_SUPPKEY`, 1) = COALESCE(`t21`.`S_SUPPKEY`, 1)
                        ) AS `t27`
                      ) AS `t29`
                      INNER JOIN (
                        SELECT
                          `t4`.`O_ORDERKEY`,
                          `t4`.`O_CUSTKEY`,
                          `t4`.`O_ORDERDATE`
                        FROM (
                          SELECT
                            `O_ORDERKEY`,
                            `O_CUSTKEY`,
                            `O_ORDERDATE`
                          FROM `bigframes-dev.tpch.ORDERS` FOR SYSTEM_TIME AS OF CAST('2026-03-10T18:00:00' AS DATETIME)
                        ) AS `t4`
                      ) AS `t20`
                        ON COALESCE(`t29`.`L_ORDERKEY`, 0) = COALESCE(`t20`.`O_ORDERKEY`, 0)
                        AND COALESCE(`t29`.`L_ORDERKEY`, 1) = COALESCE(`t20`.`O_ORDERKEY`, 1)
                    ) AS `t30`
                  ) AS `t32`
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
                  ) AS `t19`
                    ON COALESCE(`t32`.`O_CUSTKEY`, 0) = COALESCE(`t19`.`C_CUSTKEY`, 0)
                    AND COALESCE(`t32`.`O_CUSTKEY`, 1) = COALESCE(`t19`.`C_CUSTKEY`, 1)
                ) AS `t33`
              ) AS `t35`
              INNER JOIN (
                SELECT
                  `t2`.`N_NATIONKEY`,
                  `t2`.`N_REGIONKEY`
                FROM (
                  SELECT
                    `N_NATIONKEY`,
                    `N_REGIONKEY`
                  FROM `bigframes-dev.tpch.NATION` FOR SYSTEM_TIME AS OF CAST('2026-03-10T18:00:00' AS DATETIME)
                ) AS `t2`
              ) AS `t18`
                ON COALESCE(`t35`.`C_NATIONKEY`, 0) = COALESCE(`t18`.`N_NATIONKEY`, 0)
                AND COALESCE(`t35`.`C_NATIONKEY`, 1) = COALESCE(`t18`.`N_NATIONKEY`, 1)
            ) AS `t36`
          ) AS `t38`
          INNER JOIN (
            SELECT
              `t1`.`R_REGIONKEY`,
              `t1`.`R_NAME`
            FROM (
              SELECT
                `R_REGIONKEY`,
                `R_NAME`
              FROM `bigframes-dev.tpch.REGION` FOR SYSTEM_TIME AS OF CAST('2026-03-10T18:00:00' AS DATETIME)
            ) AS `t1`
          ) AS `t17`
            ON COALESCE(`t38`.`N_REGIONKEY`, 0) = COALESCE(`t17`.`R_REGIONKEY`, 0)
            AND COALESCE(`t38`.`N_REGIONKEY`, 1) = COALESCE(`t17`.`R_REGIONKEY`, 1)
        ) AS `t39`
        WHERE
          `t39`.`bfuid_col_2494`
      ) AS `t41`
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
      ) AS `t16`
        ON COALESCE(`t41`.`bfuid_col_2468`, 0) = COALESCE(`t16`.`N_NATIONKEY`, 0)
        AND COALESCE(`t41`.`bfuid_col_2468`, 1) = COALESCE(`t16`.`N_NATIONKEY`, 1)
    ) AS `t42`
    WHERE
      `t42`.`bfuid_col_2556` AND `t42`.`bfuid_col_2504` = 'ECONOMY ANODIZED STEEL'
  ) AS `t43`
  GROUP BY
    1
) AS `t44`
WHERE
  (
    `t44`.`bfuid_col_2854`
  ) IS NOT NULL) AS `t`
ORDER BY `O_YEAR` ASC NULLS LAST ,`bfuid_col_2870` ASC NULLS LAST