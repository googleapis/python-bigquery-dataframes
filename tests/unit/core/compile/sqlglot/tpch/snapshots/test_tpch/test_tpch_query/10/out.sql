SELECT `C_CUSTKEY`, `C_NAME`, `REVENUE`, `C_ACCTBAL`, `N_NAME`, `C_ADDRESS`, `C_PHONE`, `C_COMMENT` FROM (SELECT
  `t20`.`bfuid_col_3099` AS `C_CUSTKEY`,
  `t20`.`bfuid_col_3100` AS `C_NAME`,
  `t20`.`bfuid_col_3137` AS `REVENUE`,
  `t20`.`bfuid_col_3104` AS `C_ACCTBAL`,
  `t20`.`bfuid_col_3133` AS `N_NAME`,
  `t20`.`bfuid_col_3101` AS `C_ADDRESS`,
  `t20`.`bfuid_col_3103` AS `C_PHONE`,
  `t20`.`bfuid_col_3106` AS `C_COMMENT`
FROM (
  SELECT
    `t19`.`bfuid_col_3099`,
    `t19`.`bfuid_col_3100`,
    `t19`.`bfuid_col_3104`,
    `t19`.`bfuid_col_3103`,
    `t19`.`bfuid_col_3133`,
    `t19`.`bfuid_col_3101`,
    `t19`.`bfuid_col_3106`,
    COALESCE(SUM(`t19`.`bfuid_col_3136`), 0) AS `bfuid_col_3137`
  FROM (
    SELECT
      `t18`.`bfuid_col_3056` AS `bfuid_col_3099`,
      `t18`.`bfuid_col_3057` AS `bfuid_col_3100`,
      `t18`.`bfuid_col_3058` AS `bfuid_col_3101`,
      `t18`.`bfuid_col_3060` AS `bfuid_col_3103`,
      `t18`.`bfuid_col_3061` AS `bfuid_col_3104`,
      `t18`.`bfuid_col_3063` AS `bfuid_col_3106`,
      `t18`.`bfuid_col_3090` AS `bfuid_col_3133`,
      ROUND(`t18`.`bfuid_col_3078` * (
        1 - `t18`.`bfuid_col_3079`
      ), 2) AS `bfuid_col_3136`
    FROM (
      SELECT
        `t17`.`C_CUSTKEY` AS `bfuid_col_3056`,
        `t17`.`C_NAME` AS `bfuid_col_3057`,
        `t17`.`C_ADDRESS` AS `bfuid_col_3058`,
        `t17`.`C_PHONE` AS `bfuid_col_3060`,
        `t17`.`C_ACCTBAL` AS `bfuid_col_3061`,
        `t17`.`C_COMMENT` AS `bfuid_col_3063`,
        `t17`.`L_EXTENDEDPRICE` AS `bfuid_col_3078`,
        `t17`.`L_DISCOUNT` AS `bfuid_col_3079`,
        `t8`.`N_NAME` AS `bfuid_col_3090`,
        (
          (
            `t17`.`O_ORDERDATE` >= DATE(1993, 10, 1)
          )
          AND (
            `t17`.`O_ORDERDATE` < DATE(1994, 1, 1)
          )
        )
        AND (
          `t17`.`L_RETURNFLAG` = 'R'
        ) AS `bfuid_col_3093`
      FROM (
        SELECT
          *
        FROM (
          SELECT
            `t14`.`C_CUSTKEY`,
            `t14`.`C_NAME`,
            `t14`.`C_ADDRESS`,
            `t14`.`C_NATIONKEY`,
            `t14`.`C_PHONE`,
            `t14`.`C_ACCTBAL`,
            `t14`.`C_COMMENT`,
            `t14`.`O_ORDERDATE`,
            `t9`.`L_EXTENDEDPRICE`,
            `t9`.`L_DISCOUNT`,
            `t9`.`L_RETURNFLAG`
          FROM (
            SELECT
              *
            FROM (
              SELECT
                `t10`.`C_CUSTKEY`,
                `t10`.`C_NAME`,
                `t10`.`C_ADDRESS`,
                `t10`.`C_NATIONKEY`,
                `t10`.`C_PHONE`,
                `t10`.`C_ACCTBAL`,
                `t10`.`C_COMMENT`,
                `t11`.`O_ORDERKEY`,
                `t11`.`O_ORDERDATE`
              FROM (
                SELECT
                  `t2`.`C_CUSTKEY`,
                  `t2`.`C_NAME`,
                  `t2`.`C_ADDRESS`,
                  `t2`.`C_NATIONKEY`,
                  `t2`.`C_PHONE`,
                  `t2`.`C_ACCTBAL`,
                  `t2`.`C_COMMENT`
                FROM (
                  SELECT
                    `C_CUSTKEY`,
                    `C_NAME`,
                    `C_ADDRESS`,
                    `C_NATIONKEY`,
                    `C_PHONE`,
                    `C_ACCTBAL`,
                    `C_COMMENT`
                  FROM `bigframes-dev.tpch.CUSTOMER` FOR SYSTEM_TIME AS OF CAST('2026-03-10T18:00:00' AS DATETIME)
                ) AS `t2`
              ) AS `t10`
              INNER JOIN (
                SELECT
                  `t3`.`O_ORDERKEY`,
                  `t3`.`O_CUSTKEY`,
                  `t3`.`O_ORDERDATE`
                FROM (
                  SELECT
                    `O_ORDERKEY`,
                    `O_CUSTKEY`,
                    `O_ORDERDATE`
                  FROM `bigframes-dev.tpch.ORDERS` FOR SYSTEM_TIME AS OF CAST('2026-03-10T18:00:00' AS DATETIME)
                ) AS `t3`
              ) AS `t11`
                ON COALESCE(`t10`.`C_CUSTKEY`, 0) = COALESCE(`t11`.`O_CUSTKEY`, 0)
                AND COALESCE(`t10`.`C_CUSTKEY`, 1) = COALESCE(`t11`.`O_CUSTKEY`, 1)
            ) AS `t12`
          ) AS `t14`
          INNER JOIN (
            SELECT
              `t1`.`L_ORDERKEY`,
              `t1`.`L_EXTENDEDPRICE`,
              `t1`.`L_DISCOUNT`,
              `t1`.`L_RETURNFLAG`
            FROM (
              SELECT
                `L_ORDERKEY`,
                `L_EXTENDEDPRICE`,
                `L_DISCOUNT`,
                `L_RETURNFLAG`
              FROM `bigframes-dev.tpch.LINEITEM` FOR SYSTEM_TIME AS OF CAST('2026-03-10T18:00:00' AS DATETIME)
            ) AS `t1`
          ) AS `t9`
            ON COALESCE(`t14`.`O_ORDERKEY`, 0) = COALESCE(`t9`.`L_ORDERKEY`, 0)
            AND COALESCE(`t14`.`O_ORDERKEY`, 1) = COALESCE(`t9`.`L_ORDERKEY`, 1)
        ) AS `t15`
      ) AS `t17`
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
      ) AS `t8`
        ON COALESCE(`t17`.`C_NATIONKEY`, 0) = COALESCE(`t8`.`N_NATIONKEY`, 0)
        AND COALESCE(`t17`.`C_NATIONKEY`, 1) = COALESCE(`t8`.`N_NATIONKEY`, 1)
    ) AS `t18`
    WHERE
      `t18`.`bfuid_col_3093`
  ) AS `t19`
  GROUP BY
    1,
    2,
    3,
    4,
    5,
    6,
    7
) AS `t20`
WHERE
  (
    `t20`.`bfuid_col_3099`
  ) IS NOT NULL
  AND (
    `t20`.`bfuid_col_3100`
  ) IS NOT NULL
  AND (
    `t20`.`bfuid_col_3104`
  ) IS NOT NULL
  AND (
    `t20`.`bfuid_col_3103`
  ) IS NOT NULL
  AND (
    `t20`.`bfuid_col_3133`
  ) IS NOT NULL
  AND (
    `t20`.`bfuid_col_3101`
  ) IS NOT NULL
  AND (
    `t20`.`bfuid_col_3106`
  ) IS NOT NULL) AS `t`
ORDER BY `REVENUE` DESC NULLS LAST ,`C_CUSTKEY` ASC NULLS LAST ,`C_NAME` ASC NULLS LAST ,`C_ACCTBAL` ASC NULLS LAST ,`C_PHONE` ASC NULLS LAST ,`N_NAME` ASC NULLS LAST ,`C_ADDRESS` ASC NULLS LAST ,`C_COMMENT` ASC NULLS LAST
LIMIT 20