SELECT `O_ORDERPRIORITY`, `ORDER_COUNT` FROM (SELECT
  `t10`.`bfuid_col_2134` AS `O_ORDERPRIORITY`,
  `t10`.`bfuid_col_2140` AS `ORDER_COUNT`
FROM (
  SELECT
    `t9`.`bfuid_col_2134`,
    COUNT(`t9`.`bfuid_col_2113`) AS `bfuid_col_2140`
  FROM (
    SELECT
      `t8`.`bfuid_col_2134`,
      `t8`.`bfuid_col_2113`
    FROM (
      SELECT
        `t7`.`bfuid_col_2134`,
        `t7`.`bfuid_col_2113`,
        COUNT(1) AS `bfuid_col_2139`
      FROM (
        SELECT
          `t6`.`bfuid_col_2084` AS `bfuid_col_2113`,
          `t6`.`bfuid_col_2105` AS `bfuid_col_2134`
        FROM (
          SELECT
            `t4`.`L_ORDERKEY` AS `bfuid_col_2084`,
            `t4`.`L_COMMITDATE` AS `bfuid_col_2095`,
            `t4`.`L_RECEIPTDATE` AS `bfuid_col_2096`,
            `t5`.`O_ORDERPRIORITY` AS `bfuid_col_2105`,
            (
              `t5`.`O_ORDERDATE` >= DATE(1993, 7, 1)
            )
            AND (
              `t5`.`O_ORDERDATE` < DATE(1993, 10, 1)
            ) AS `bfuid_col_2109`
          FROM (
            SELECT
              `t0`.`L_ORDERKEY`,
              `t0`.`L_COMMITDATE`,
              `t0`.`L_RECEIPTDATE`
            FROM (
              SELECT
                `L_ORDERKEY`,
                `L_COMMITDATE`,
                `L_RECEIPTDATE`
              FROM `bigframes-dev.tpch.LINEITEM` FOR SYSTEM_TIME AS OF CAST('2026-03-10T18:00:00' AS DATETIME)
            ) AS `t0`
          ) AS `t4`
          INNER JOIN (
            SELECT
              `t1`.`O_ORDERKEY`,
              `t1`.`O_ORDERDATE`,
              `t1`.`O_ORDERPRIORITY`
            FROM (
              SELECT
                `O_ORDERKEY`,
                `O_ORDERDATE`,
                `O_ORDERPRIORITY`
              FROM `bigframes-dev.tpch.ORDERS` FOR SYSTEM_TIME AS OF CAST('2026-03-10T18:00:00' AS DATETIME)
            ) AS `t1`
          ) AS `t5`
            ON COALESCE(`t4`.`L_ORDERKEY`, 0) = COALESCE(`t5`.`O_ORDERKEY`, 0)
            AND COALESCE(`t4`.`L_ORDERKEY`, 1) = COALESCE(`t5`.`O_ORDERKEY`, 1)
        ) AS `t6`
        WHERE
          `t6`.`bfuid_col_2109` AND `t6`.`bfuid_col_2095` < `t6`.`bfuid_col_2096`
      ) AS `t7`
      GROUP BY
        1,
        2
    ) AS `t8`
    WHERE
      (
        `t8`.`bfuid_col_2134`
      ) IS NOT NULL
      AND (
        `t8`.`bfuid_col_2113`
      ) IS NOT NULL
  ) AS `t9`
  GROUP BY
    1
) AS `t10`) AS `t`
ORDER BY `O_ORDERPRIORITY` ASC NULLS LAST