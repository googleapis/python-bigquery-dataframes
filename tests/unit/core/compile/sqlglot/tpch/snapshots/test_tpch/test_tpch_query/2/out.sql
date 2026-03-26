SELECT `S_ACCTBAL`, `S_NAME`, `N_NAME`, `P_PARTKEY`, `P_MFGR`, `S_ADDRESS`, `S_PHONE`, `S_COMMENT` FROM (SELECT
  `t46`.`bfuid_col_1989` AS `S_ACCTBAL`,
  `t46`.`bfuid_col_1985` AS `S_NAME`,
  `t46`.`bfuid_col_1992` AS `N_NAME`,
  `t49`.`bfuid_col_1970` AS `P_PARTKEY`,
  `t46`.`bfuid_col_1972` AS `P_MFGR`,
  `t46`.`bfuid_col_1986` AS `S_ADDRESS`,
  `t46`.`bfuid_col_1988` AS `S_PHONE`,
  `t46`.`bfuid_col_1990` AS `S_COMMENT`
FROM (
  SELECT
    *
  FROM (
    SELECT
      `t45`.`bfuid_col_1970`,
      MIN(`t45`.`bfuid_col_1982`) AS `bfuid_col_1999`
    FROM (
      SELECT
        `t43`.`bfuid_col_1910` AS `bfuid_col_1970`,
        `t43`.`bfuid_col_1922` AS `bfuid_col_1982`
      FROM (
        SELECT
          `t41`.`P_PARTKEY` AS `bfuid_col_1910`,
          `t41`.`P_TYPE` AS `bfuid_col_1914`,
          `t41`.`PS_SUPPLYCOST` AS `bfuid_col_1922`,
          `t16`.`R_NAME` AS `bfuid_col_1936`,
          `t41`.`P_SIZE` = 15 AS `bfuid_col_1938`
        FROM (
          SELECT
            *
          FROM (
            SELECT
              `t35`.`P_PARTKEY`,
              `t35`.`P_TYPE`,
              `t35`.`P_SIZE`,
              `t35`.`PS_SUPPLYCOST`,
              `t18`.`N_REGIONKEY`
            FROM (
              SELECT
                *
              FROM (
                SELECT
                  `t29`.`P_PARTKEY`,
                  `t29`.`P_TYPE`,
                  `t29`.`P_SIZE`,
                  `t29`.`PS_SUPPLYCOST`,
                  `t20`.`S_NATIONKEY`
                FROM (
                  SELECT
                    *
                  FROM (
                    SELECT
                      `t22`.`P_PARTKEY`,
                      `t22`.`P_TYPE`,
                      `t22`.`P_SIZE`,
                      `t23`.`PS_SUPPKEY`,
                      `t23`.`PS_SUPPLYCOST`
                    FROM (
                      SELECT
                        `t6`.`P_PARTKEY`,
                        `t6`.`P_TYPE`,
                        `t6`.`P_SIZE`
                      FROM (
                        SELECT
                          `P_PARTKEY`,
                          `P_TYPE`,
                          `P_SIZE`
                        FROM `bigframes-dev.tpch.PART` FOR SYSTEM_TIME AS OF CAST('2026-03-10T18:00:00' AS DATETIME)
                      ) AS `t6`
                    ) AS `t22`
                    INNER JOIN (
                      SELECT
                        `t7`.`PS_PARTKEY`,
                        `t7`.`PS_SUPPKEY`,
                        `t7`.`PS_SUPPLYCOST`
                      FROM (
                        SELECT
                          `PS_PARTKEY`,
                          `PS_SUPPKEY`,
                          `PS_SUPPLYCOST`
                        FROM `bigframes-dev.tpch.PARTSUPP` FOR SYSTEM_TIME AS OF CAST('2026-03-10T18:00:00' AS DATETIME)
                      ) AS `t7`
                    ) AS `t23`
                      ON COALESCE(`t22`.`P_PARTKEY`, 0) = COALESCE(`t23`.`PS_PARTKEY`, 0)
                      AND COALESCE(`t22`.`P_PARTKEY`, 1) = COALESCE(`t23`.`PS_PARTKEY`, 1)
                  ) AS `t25`
                ) AS `t29`
                INNER JOIN (
                  SELECT
                    `t4`.`S_SUPPKEY`,
                    `t4`.`S_NATIONKEY`
                  FROM (
                    SELECT
                      `S_SUPPKEY`,
                      `S_NATIONKEY`
                    FROM `bigframes-dev.tpch.SUPPLIER` FOR SYSTEM_TIME AS OF CAST('2026-03-10T18:00:00' AS DATETIME)
                  ) AS `t4`
                ) AS `t20`
                  ON COALESCE(`t29`.`PS_SUPPKEY`, 0) = COALESCE(`t20`.`S_SUPPKEY`, 0)
                  AND COALESCE(`t29`.`PS_SUPPKEY`, 1) = COALESCE(`t20`.`S_SUPPKEY`, 1)
              ) AS `t31`
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
              ON COALESCE(`t35`.`S_NATIONKEY`, 0) = COALESCE(`t18`.`N_NATIONKEY`, 0)
              AND COALESCE(`t35`.`S_NATIONKEY`, 1) = COALESCE(`t18`.`N_NATIONKEY`, 1)
          ) AS `t37`
        ) AS `t41`
        INNER JOIN (
          SELECT
            `t0`.`R_REGIONKEY`,
            `t0`.`R_NAME`
          FROM (
            SELECT
              `R_REGIONKEY`,
              `R_NAME`
            FROM `bigframes-dev.tpch.REGION` FOR SYSTEM_TIME AS OF CAST('2026-03-10T18:00:00' AS DATETIME)
          ) AS `t0`
        ) AS `t16`
          ON COALESCE(`t41`.`N_REGIONKEY`, 0) = COALESCE(`t16`.`R_REGIONKEY`, 0)
          AND COALESCE(`t41`.`N_REGIONKEY`, 1) = COALESCE(`t16`.`R_REGIONKEY`, 1)
      ) AS `t43`
      WHERE
        `t43`.`bfuid_col_1938`
        AND ENDS_WITH(`t43`.`bfuid_col_1914`, 'BRASS')
        AND `t43`.`bfuid_col_1936` = 'EUROPE'
    ) AS `t45`
    GROUP BY
      1
  ) AS `t47`
  WHERE
    (
      `t47`.`bfuid_col_1970`
    ) IS NOT NULL
) AS `t49`
INNER JOIN (
  SELECT
    `t42`.`bfuid_col_1910` AS `bfuid_col_2000`,
    `t42`.`bfuid_col_1912` AS `bfuid_col_1972`,
    `t42`.`bfuid_col_1922` AS `bfuid_col_1982`,
    `t42`.`bfuid_col_1925` AS `bfuid_col_1985`,
    `t42`.`bfuid_col_1926` AS `bfuid_col_1986`,
    `t42`.`bfuid_col_1928` AS `bfuid_col_1988`,
    `t42`.`bfuid_col_1929` AS `bfuid_col_1989`,
    `t42`.`bfuid_col_1930` AS `bfuid_col_1990`,
    `t42`.`bfuid_col_1932` AS `bfuid_col_1992`
  FROM (
    SELECT
      `t40`.`P_PARTKEY` AS `bfuid_col_1910`,
      `t40`.`P_MFGR` AS `bfuid_col_1912`,
      `t40`.`P_TYPE` AS `bfuid_col_1914`,
      `t40`.`PS_SUPPLYCOST` AS `bfuid_col_1922`,
      `t40`.`S_NAME` AS `bfuid_col_1925`,
      `t40`.`S_ADDRESS` AS `bfuid_col_1926`,
      `t40`.`S_PHONE` AS `bfuid_col_1928`,
      `t40`.`S_ACCTBAL` AS `bfuid_col_1929`,
      `t40`.`S_COMMENT` AS `bfuid_col_1930`,
      `t40`.`N_NAME` AS `bfuid_col_1932`,
      `t16`.`R_NAME` AS `bfuid_col_1936`,
      `t40`.`P_SIZE` = 15 AS `bfuid_col_1938`
    FROM (
      SELECT
        *
      FROM (
        SELECT
          `t34`.`P_PARTKEY`,
          `t34`.`P_MFGR`,
          `t34`.`P_TYPE`,
          `t34`.`P_SIZE`,
          `t34`.`PS_SUPPLYCOST`,
          `t34`.`S_NAME`,
          `t34`.`S_ADDRESS`,
          `t34`.`S_PHONE`,
          `t34`.`S_ACCTBAL`,
          `t34`.`S_COMMENT`,
          `t17`.`N_NAME`,
          `t17`.`N_REGIONKEY`
        FROM (
          SELECT
            *
          FROM (
            SELECT
              `t28`.`P_PARTKEY`,
              `t28`.`P_MFGR`,
              `t28`.`P_TYPE`,
              `t28`.`P_SIZE`,
              `t28`.`PS_SUPPLYCOST`,
              `t19`.`S_NAME`,
              `t19`.`S_ADDRESS`,
              `t19`.`S_NATIONKEY`,
              `t19`.`S_PHONE`,
              `t19`.`S_ACCTBAL`,
              `t19`.`S_COMMENT`
            FROM (
              SELECT
                *
              FROM (
                SELECT
                  `t21`.`P_PARTKEY`,
                  `t21`.`P_MFGR`,
                  `t21`.`P_TYPE`,
                  `t21`.`P_SIZE`,
                  `t23`.`PS_SUPPKEY`,
                  `t23`.`PS_SUPPLYCOST`
                FROM (
                  SELECT
                    `t5`.`P_PARTKEY`,
                    `t5`.`P_MFGR`,
                    `t5`.`P_TYPE`,
                    `t5`.`P_SIZE`
                  FROM (
                    SELECT
                      `P_PARTKEY`,
                      `P_MFGR`,
                      `P_TYPE`,
                      `P_SIZE`
                    FROM `bigframes-dev.tpch.PART` FOR SYSTEM_TIME AS OF CAST('2026-03-10T18:00:00' AS DATETIME)
                  ) AS `t5`
                ) AS `t21`
                INNER JOIN (
                  SELECT
                    `t7`.`PS_PARTKEY`,
                    `t7`.`PS_SUPPKEY`,
                    `t7`.`PS_SUPPLYCOST`
                  FROM (
                    SELECT
                      `PS_PARTKEY`,
                      `PS_SUPPKEY`,
                      `PS_SUPPLYCOST`
                    FROM `bigframes-dev.tpch.PARTSUPP` FOR SYSTEM_TIME AS OF CAST('2026-03-10T18:00:00' AS DATETIME)
                  ) AS `t7`
                ) AS `t23`
                  ON COALESCE(`t21`.`P_PARTKEY`, 0) = COALESCE(`t23`.`PS_PARTKEY`, 0)
                  AND COALESCE(`t21`.`P_PARTKEY`, 1) = COALESCE(`t23`.`PS_PARTKEY`, 1)
              ) AS `t24`
            ) AS `t28`
            INNER JOIN (
              SELECT
                *
              FROM (
                SELECT
                  `S_SUPPKEY`,
                  `S_NAME`,
                  `S_ADDRESS`,
                  `S_NATIONKEY`,
                  `S_PHONE`,
                  `S_ACCTBAL`,
                  `S_COMMENT`
                FROM `bigframes-dev.tpch.SUPPLIER` FOR SYSTEM_TIME AS OF CAST('2026-03-10T18:00:00' AS DATETIME)
              ) AS `t3`
            ) AS `t19`
              ON COALESCE(`t28`.`PS_SUPPKEY`, 0) = COALESCE(`t19`.`S_SUPPKEY`, 0)
              AND COALESCE(`t28`.`PS_SUPPKEY`, 1) = COALESCE(`t19`.`S_SUPPKEY`, 1)
          ) AS `t30`
        ) AS `t34`
        INNER JOIN (
          SELECT
            `t1`.`N_NATIONKEY`,
            `t1`.`N_NAME`,
            `t1`.`N_REGIONKEY`
          FROM (
            SELECT
              `N_NATIONKEY`,
              `N_NAME`,
              `N_REGIONKEY`
            FROM `bigframes-dev.tpch.NATION` FOR SYSTEM_TIME AS OF CAST('2026-03-10T18:00:00' AS DATETIME)
          ) AS `t1`
        ) AS `t17`
          ON COALESCE(`t34`.`S_NATIONKEY`, 0) = COALESCE(`t17`.`N_NATIONKEY`, 0)
          AND COALESCE(`t34`.`S_NATIONKEY`, 1) = COALESCE(`t17`.`N_NATIONKEY`, 1)
      ) AS `t36`
    ) AS `t40`
    INNER JOIN (
      SELECT
        `t0`.`R_REGIONKEY`,
        `t0`.`R_NAME`
      FROM (
        SELECT
          `R_REGIONKEY`,
          `R_NAME`
        FROM `bigframes-dev.tpch.REGION` FOR SYSTEM_TIME AS OF CAST('2026-03-10T18:00:00' AS DATETIME)
      ) AS `t0`
    ) AS `t16`
      ON COALESCE(`t40`.`N_REGIONKEY`, 0) = COALESCE(`t16`.`R_REGIONKEY`, 0)
      AND COALESCE(`t40`.`N_REGIONKEY`, 1) = COALESCE(`t16`.`R_REGIONKEY`, 1)
  ) AS `t42`
  WHERE
    `t42`.`bfuid_col_1938`
    AND ENDS_WITH(`t42`.`bfuid_col_1914`, 'BRASS')
    AND `t42`.`bfuid_col_1936` = 'EUROPE'
) AS `t46`
  ON COALESCE(`t49`.`bfuid_col_1970`, 0) = COALESCE(`t46`.`bfuid_col_2000`, 0)
  AND COALESCE(`t49`.`bfuid_col_1970`, 1) = COALESCE(`t46`.`bfuid_col_2000`, 1)
  AND IF(IS_NAN(`t49`.`bfuid_col_1999`), 2, COALESCE(`t49`.`bfuid_col_1999`, 0)) = IF(IS_NAN(`t46`.`bfuid_col_1982`), 2, COALESCE(`t46`.`bfuid_col_1982`, 0))
  AND IF(IS_NAN(`t49`.`bfuid_col_1999`), 3, COALESCE(`t49`.`bfuid_col_1999`, 1)) = IF(IS_NAN(`t46`.`bfuid_col_1982`), 3, COALESCE(`t46`.`bfuid_col_1982`, 1))) AS `t`
ORDER BY `S_ACCTBAL` DESC NULLS LAST ,`N_NAME` ASC NULLS LAST ,`S_NAME` ASC NULLS LAST ,`P_PARTKEY` ASC NULLS LAST
LIMIT 100