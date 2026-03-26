SELECT `L_ORDERKEY`, `REVENUE`, `O_ORDERDATE`, `O_SHIPPRIORITY` FROM (SELECT
  `t13`.`bfuid_col_2068` AS `L_ORDERKEY`,
  `t13`.`bfuid_col_2078` AS `REVENUE`,
  `t13`.`bfuid_col_2072` AS `O_ORDERDATE`,
  `t13`.`bfuid_col_2075` AS `O_SHIPPRIORITY`
FROM (
  SELECT
    `t12`.`bfuid_col_2068`,
    `t12`.`bfuid_col_2072`,
    `t12`.`bfuid_col_2075`,
    COALESCE(SUM(`t12`.`bfuid_col_2077`), 0) AS `bfuid_col_2078`
  FROM (
    SELECT
      `t11`.`bfuid_col_2012` AS `bfuid_col_2068`,
      `t11`.`bfuid_col_2016` AS `bfuid_col_2072`,
      `t11`.`bfuid_col_2019` AS `bfuid_col_2075`,
      `t11`.`bfuid_col_2028` * (
        1 - `t11`.`bfuid_col_2029`
      ) AS `bfuid_col_2077`
    FROM (
      SELECT
        `t0`.`C_CUSTKEY` AS `bfuid_col_2002`
      FROM (
        SELECT
          `C_CUSTKEY`,
          `C_MKTSEGMENT`
        FROM `bigframes-dev.tpch.CUSTOMER` FOR SYSTEM_TIME AS OF CAST('2026-03-10T18:00:00' AS DATETIME)
      ) AS `t0`
      WHERE
        `t0`.`C_MKTSEGMENT` = 'BUILDING'
    ) AS `t6`
    INNER JOIN (
      SELECT
        *
      FROM (
        SELECT
          `t7`.`bfuid_col_2028`,
          `t7`.`bfuid_col_2029`,
          `t8`.`bfuid_col_2012`,
          `t8`.`bfuid_col_2013`,
          `t8`.`bfuid_col_2016`,
          `t8`.`bfuid_col_2019`
        FROM (
          SELECT
            `t1`.`L_ORDERKEY` AS `bfuid_col_2023`,
            `t1`.`L_EXTENDEDPRICE` AS `bfuid_col_2028`,
            `t1`.`L_DISCOUNT` AS `bfuid_col_2029`
          FROM (
            SELECT
              `L_ORDERKEY`,
              `L_EXTENDEDPRICE`,
              `L_DISCOUNT`,
              `L_SHIPDATE`
            FROM `bigframes-dev.tpch.LINEITEM` FOR SYSTEM_TIME AS OF CAST('2026-03-10T18:00:00' AS DATETIME)
          ) AS `t1`
          WHERE
            `t1`.`L_SHIPDATE` > DATE(1995, 3, 15)
        ) AS `t7`
        INNER JOIN (
          SELECT
            `t2`.`O_ORDERKEY` AS `bfuid_col_2012`,
            `t2`.`O_CUSTKEY` AS `bfuid_col_2013`,
            `t2`.`O_ORDERDATE` AS `bfuid_col_2016`,
            `t2`.`O_SHIPPRIORITY` AS `bfuid_col_2019`
          FROM (
            SELECT
              `O_ORDERKEY`,
              `O_CUSTKEY`,
              `O_ORDERDATE`,
              `O_SHIPPRIORITY`
            FROM `bigframes-dev.tpch.ORDERS` FOR SYSTEM_TIME AS OF CAST('2026-03-10T18:00:00' AS DATETIME)
          ) AS `t2`
          WHERE
            `t2`.`O_ORDERDATE` < DATE(1995, 3, 15)
        ) AS `t8`
          ON COALESCE(`t7`.`bfuid_col_2023`, 0) = COALESCE(`t8`.`bfuid_col_2012`, 0)
          AND COALESCE(`t7`.`bfuid_col_2023`, 1) = COALESCE(`t8`.`bfuid_col_2012`, 1)
      ) AS `t9`
    ) AS `t11`
      ON COALESCE(`t6`.`bfuid_col_2002`, 0) = COALESCE(`t11`.`bfuid_col_2013`, 0)
      AND COALESCE(`t6`.`bfuid_col_2002`, 1) = COALESCE(`t11`.`bfuid_col_2013`, 1)
  ) AS `t12`
  GROUP BY
    1,
    2,
    3
) AS `t13`
WHERE
  (
    `t13`.`bfuid_col_2068`
  ) IS NOT NULL
  AND (
    `t13`.`bfuid_col_2072`
  ) IS NOT NULL
  AND (
    `t13`.`bfuid_col_2075`
  ) IS NOT NULL) AS `t`
ORDER BY `REVENUE` DESC NULLS LAST ,`O_ORDERDATE` ASC NULLS LAST ,`L_ORDERKEY` ASC NULLS LAST ,`O_SHIPPRIORITY` ASC NULLS LAST
LIMIT 10