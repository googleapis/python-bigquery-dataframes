SELECT `SUPP_NATION`, `CUST_NATION`, `L_YEAR`, `REVENUE` FROM (WITH `t12` AS (
  SELECT
    `t0`.`N_NATIONKEY` AS `bfuid_col_2251`,
    `t0`.`N_NAME` AS `bfuid_col_2252`,
    COALESCE(COALESCE(`t0`.`N_NAME` IN ('FRANCE', 'GERMANY'), FALSE), FALSE) AS `bfuid_col_2255`
  FROM (
    SELECT
      `N_NATIONKEY`,
      `N_NAME`
    FROM `bigframes-dev.tpch.NATION` FOR SYSTEM_TIME AS OF CAST('2026-03-10T18:00:00' AS DATETIME)
  ) AS `t0`
)
SELECT
  `t33`.`bfuid_col_2433` AS `SUPP_NATION`,
  `t33`.`bfuid_col_2397` AS `CUST_NATION`,
  `t33`.`bfuid_col_2437` AS `L_YEAR`,
  `t33`.`bfuid_col_2438` AS `REVENUE`
FROM (
  SELECT
    `t32`.`bfuid_col_2433`,
    `t32`.`bfuid_col_2397`,
    `t32`.`bfuid_col_2437`,
    COALESCE(SUM(`t32`.`bfuid_col_2436`), 0) AS `bfuid_col_2438`
  FROM (
    SELECT
      `t31`.`bfuid_col_2294` AS `bfuid_col_2397`,
      `t31`.`bfuid_col_2330` AS `bfuid_col_2433`,
      `t31`.`bfuid_col_2311` * (
        1.0 - `t31`.`bfuid_col_2312`
      ) AS `bfuid_col_2436`,
      EXTRACT(year FROM `t31`.`bfuid_col_2316`) AS `bfuid_col_2437`
    FROM (
      SELECT
        `t30`.`bfuid_col_2252` AS `bfuid_col_2294`,
        `t30`.`bfuid_col_2266` AS `bfuid_col_2311`,
        `t30`.`bfuid_col_2267` AS `bfuid_col_2312`,
        `t30`.`bfuid_col_2271` AS `bfuid_col_2316`,
        `t17`.`bfuid_col_2279` AS `bfuid_col_2330`,
        `t30`.`bfuid_col_2252` <> `t17`.`bfuid_col_2279` AS `bfuid_col_2333`
      FROM (
        SELECT
          *
        FROM (
          SELECT
            `t27`.`bfuid_col_2252`,
            `t27`.`bfuid_col_2266`,
            `t27`.`bfuid_col_2267`,
            `t27`.`bfuid_col_2271`,
            `t8`.`S_NATIONKEY`
          FROM (
            SELECT
              *
            FROM (
              SELECT
                `t24`.`bfuid_col_2252`,
                `t13`.`bfuid_col_2263`,
                `t13`.`bfuid_col_2266`,
                `t13`.`bfuid_col_2267`,
                `t13`.`bfuid_col_2271`
              FROM (
                SELECT
                  *
                FROM (
                  SELECT
                    `t21`.`bfuid_col_2252`,
                    `t9`.`O_ORDERKEY`
                  FROM (
                    SELECT
                      *
                    FROM (
                      SELECT
                        `t10`.`C_CUSTKEY`,
                        `t18`.`bfuid_col_2252`
                      FROM (
                        SELECT
                          `t4`.`C_CUSTKEY`,
                          `t4`.`C_NATIONKEY`
                        FROM (
                          SELECT
                            `C_CUSTKEY`,
                            `C_NATIONKEY`
                          FROM `bigframes-dev.tpch.CUSTOMER` FOR SYSTEM_TIME AS OF CAST('2026-03-10T18:00:00' AS DATETIME)
                        ) AS `t4`
                      ) AS `t10`
                      INNER JOIN (
                        SELECT
                          `t14`.`bfuid_col_2251`,
                          `t14`.`bfuid_col_2252`
                        FROM `t12` AS `t14`
                        WHERE
                          `t14`.`bfuid_col_2255`
                      ) AS `t18`
                        ON COALESCE(`t10`.`C_NATIONKEY`, 0) = COALESCE(`t18`.`bfuid_col_2251`, 0)
                        AND COALESCE(`t10`.`C_NATIONKEY`, 1) = COALESCE(`t18`.`bfuid_col_2251`, 1)
                    ) AS `t19`
                  ) AS `t21`
                  INNER JOIN (
                    SELECT
                      `t3`.`O_ORDERKEY`,
                      `t3`.`O_CUSTKEY`
                    FROM (
                      SELECT
                        `O_ORDERKEY`,
                        `O_CUSTKEY`
                      FROM `bigframes-dev.tpch.ORDERS` FOR SYSTEM_TIME AS OF CAST('2026-03-10T18:00:00' AS DATETIME)
                    ) AS `t3`
                  ) AS `t9`
                    ON COALESCE(`t21`.`C_CUSTKEY`, 0) = COALESCE(`t9`.`O_CUSTKEY`, 0)
                    AND COALESCE(`t21`.`C_CUSTKEY`, 1) = COALESCE(`t9`.`O_CUSTKEY`, 1)
                ) AS `t22`
              ) AS `t24`
              INNER JOIN (
                SELECT
                  `t2`.`L_ORDERKEY` AS `bfuid_col_2261`,
                  `t2`.`L_SUPPKEY` AS `bfuid_col_2263`,
                  `t2`.`L_EXTENDEDPRICE` AS `bfuid_col_2266`,
                  `t2`.`L_DISCOUNT` AS `bfuid_col_2267`,
                  `t2`.`L_SHIPDATE` AS `bfuid_col_2271`
                FROM (
                  SELECT
                    `L_ORDERKEY`,
                    `L_SUPPKEY`,
                    `L_EXTENDEDPRICE`,
                    `L_DISCOUNT`,
                    `L_SHIPDATE`
                  FROM `bigframes-dev.tpch.LINEITEM` FOR SYSTEM_TIME AS OF CAST('2026-03-10T18:00:00' AS DATETIME)
                ) AS `t2`
                WHERE
                  (
                    `t2`.`L_SHIPDATE` >= DATE(1995, 1, 1)
                  )
                  AND (
                    `t2`.`L_SHIPDATE` <= DATE(1996, 12, 31)
                  )
              ) AS `t13`
                ON COALESCE(`t24`.`O_ORDERKEY`, 0) = COALESCE(`t13`.`bfuid_col_2261`, 0)
                AND COALESCE(`t24`.`O_ORDERKEY`, 1) = COALESCE(`t13`.`bfuid_col_2261`, 1)
            ) AS `t25`
          ) AS `t27`
          INNER JOIN (
            SELECT
              `t1`.`S_SUPPKEY`,
              `t1`.`S_NATIONKEY`
            FROM (
              SELECT
                `S_SUPPKEY`,
                `S_NATIONKEY`
              FROM `bigframes-dev.tpch.SUPPLIER` FOR SYSTEM_TIME AS OF CAST('2026-03-10T18:00:00' AS DATETIME)
            ) AS `t1`
          ) AS `t8`
            ON COALESCE(`t27`.`bfuid_col_2263`, 0) = COALESCE(`t8`.`S_SUPPKEY`, 0)
            AND COALESCE(`t27`.`bfuid_col_2263`, 1) = COALESCE(`t8`.`S_SUPPKEY`, 1)
        ) AS `t28`
      ) AS `t30`
      INNER JOIN (
        SELECT
          `t14`.`bfuid_col_2251` AS `bfuid_col_2278`,
          `t14`.`bfuid_col_2252` AS `bfuid_col_2279`
        FROM `t12` AS `t14`
        WHERE
          `t14`.`bfuid_col_2255`
      ) AS `t17`
        ON COALESCE(`t30`.`S_NATIONKEY`, 0) = COALESCE(`t17`.`bfuid_col_2278`, 0)
        AND COALESCE(`t30`.`S_NATIONKEY`, 1) = COALESCE(`t17`.`bfuid_col_2278`, 1)
    ) AS `t31`
    WHERE
      `t31`.`bfuid_col_2333`
  ) AS `t32`
  GROUP BY
    1,
    2,
    3
) AS `t33`
WHERE
  (
    `t33`.`bfuid_col_2433`
  ) IS NOT NULL
  AND (
    `t33`.`bfuid_col_2397`
  ) IS NOT NULL
  AND (
    `t33`.`bfuid_col_2437`
  ) IS NOT NULL) AS `t`
ORDER BY `SUPP_NATION` ASC NULLS LAST ,`CUST_NATION` ASC NULLS LAST ,`L_YEAR` ASC NULLS LAST