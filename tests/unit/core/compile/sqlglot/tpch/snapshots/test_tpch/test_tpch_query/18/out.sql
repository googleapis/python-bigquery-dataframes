SELECT `C_NAME`, `C_CUSTKEY`, `O_ORDERKEY`, `O_ORDERDAT`, `O_TOTALPRICE`, `COL6` FROM (WITH `t5` AS (
  SELECT
    `t2`.`L_ORDERKEY`,
    `t2`.`L_QUANTITY`
  FROM (
    SELECT
      `L_ORDERKEY`,
      `L_QUANTITY`
    FROM `bigframes-dev.tpch.LINEITEM` FOR SYSTEM_TIME AS OF CAST('2026-03-10T18:00:00' AS DATETIME)
  ) AS `t2`
)
SELECT
  `t20`.`C_NAME`,
  `t20`.`C_CUSTKEY`,
  `t20`.`O_ORDERKEY`,
  `t20`.`O_ORDERDATE` AS `O_ORDERDAT`,
  `t20`.`O_TOTALPRICE`,
  `t20`.`bfuid_col_3639` AS `COL6`
FROM (
  SELECT
    `t19`.`C_NAME`,
    `t19`.`C_CUSTKEY`,
    `t19`.`O_ORDERKEY`,
    `t19`.`O_ORDERDATE`,
    `t19`.`O_TOTALPRICE`,
    COALESCE(SUM(`t19`.`L_QUANTITY`), 0) AS `bfuid_col_3639`
  FROM (
    SELECT
      `t18`.`O_ORDERKEY`,
      `t18`.`O_TOTALPRICE`,
      `t18`.`O_ORDERDATE`,
      `t18`.`L_QUANTITY`,
      `t6`.`C_CUSTKEY`,
      `t6`.`C_NAME`
    FROM (
      SELECT
        *
      FROM (
        SELECT
          `t15`.`O_ORDERKEY`,
          `t15`.`O_CUSTKEY`,
          `t15`.`O_TOTALPRICE`,
          `t15`.`O_ORDERDATE`,
          `t8`.`L_QUANTITY`
        FROM (
          SELECT
            `t13`.`O_ORDERKEY`,
            `t13`.`O_CUSTKEY`,
            `t13`.`O_TOTALPRICE`,
            `t13`.`O_ORDERDATE`
          FROM (
            SELECT
              `t4`.`O_ORDERKEY`,
              `t4`.`O_CUSTKEY`,
              `t4`.`O_TOTALPRICE`,
              `t4`.`O_ORDERDATE`,
              EXISTS(
                SELECT
                  1
                FROM (
                  SELECT
                    `t10`.`bfuid_col_3624`
                  FROM (
                    SELECT
                      `t9`.`L_ORDERKEY` AS `bfuid_col_3624`
                    FROM (
                      SELECT
                        `t7`.`L_ORDERKEY`,
                        COALESCE(SUM(`t7`.`L_QUANTITY`), 0) AS `bfuid_col_3622`
                      FROM `t5` AS `t7`
                      GROUP BY
                        1
                    ) AS `t9`
                    WHERE
                      (
                        `t9`.`L_ORDERKEY`
                      ) IS NOT NULL AND `t9`.`bfuid_col_3622` > 300
                  ) AS `t10`
                  GROUP BY
                    1
                ) AS `t11`
                WHERE
                  (
                    COALESCE(`t4`.`O_ORDERKEY`, 0) = COALESCE(`t11`.`bfuid_col_3624`, 0)
                  )
                  AND (
                    COALESCE(`t4`.`O_ORDERKEY`, 1) = COALESCE(`t11`.`bfuid_col_3624`, 1)
                  )
              ) AS `bfuid_col_3627`
            FROM (
              SELECT
                `t1`.`O_ORDERKEY`,
                `t1`.`O_CUSTKEY`,
                `t1`.`O_TOTALPRICE`,
                `t1`.`O_ORDERDATE`
              FROM (
                SELECT
                  `O_ORDERKEY`,
                  `O_CUSTKEY`,
                  `O_TOTALPRICE`,
                  `O_ORDERDATE`
                FROM `bigframes-dev.tpch.ORDERS` FOR SYSTEM_TIME AS OF CAST('2026-03-10T18:00:00' AS DATETIME)
              ) AS `t1`
            ) AS `t4`
          ) AS `t13`
          WHERE
            `t13`.`bfuid_col_3627`
        ) AS `t15`
        INNER JOIN `t5` AS `t8`
          ON COALESCE(`t15`.`O_ORDERKEY`, 0) = COALESCE(`t8`.`L_ORDERKEY`, 0)
          AND COALESCE(`t15`.`O_ORDERKEY`, 1) = COALESCE(`t8`.`L_ORDERKEY`, 1)
      ) AS `t16`
    ) AS `t18`
    INNER JOIN (
      SELECT
        `t0`.`C_CUSTKEY`,
        `t0`.`C_NAME`
      FROM (
        SELECT
          `C_CUSTKEY`,
          `C_NAME`
        FROM `bigframes-dev.tpch.CUSTOMER` FOR SYSTEM_TIME AS OF CAST('2026-03-10T18:00:00' AS DATETIME)
      ) AS `t0`
    ) AS `t6`
      ON COALESCE(`t18`.`O_CUSTKEY`, 0) = COALESCE(`t6`.`C_CUSTKEY`, 0)
      AND COALESCE(`t18`.`O_CUSTKEY`, 1) = COALESCE(`t6`.`C_CUSTKEY`, 1)
  ) AS `t19`
  GROUP BY
    1,
    2,
    3,
    4,
    5
) AS `t20`
WHERE
  (
    `t20`.`C_NAME`
  ) IS NOT NULL
  AND (
    `t20`.`C_CUSTKEY`
  ) IS NOT NULL
  AND (
    `t20`.`O_ORDERKEY`
  ) IS NOT NULL
  AND (
    `t20`.`O_ORDERDATE`
  ) IS NOT NULL
  AND (
    `t20`.`O_TOTALPRICE`
  ) IS NOT NULL) AS `t`
ORDER BY `O_TOTALPRICE` DESC NULLS LAST ,`O_ORDERDAT` ASC NULLS LAST ,`C_NAME` ASC NULLS LAST ,`C_CUSTKEY` ASC NULLS LAST ,`O_ORDERKEY` ASC NULLS LAST
LIMIT 100