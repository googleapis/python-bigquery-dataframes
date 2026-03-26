SELECT `C_COUNT`, `CUSTDIST` FROM (SELECT
  `t9`.`bfuid_col_3309` AS `C_COUNT`,
  `t9`.`bfuid_col_3310` AS `CUSTDIST`
FROM (
  SELECT
    `t8`.`bfuid_col_3309`,
    COUNT(1) AS `bfuid_col_3310`
  FROM (
    SELECT
      `t7`.`bfuid_col_3309`
    FROM (
      SELECT
        `t6`.`C_CUSTKEY`,
        COUNT(`t6`.`bfuid_col_3299`) AS `bfuid_col_3309`
      FROM (
        SELECT
          `t3`.`C_CUSTKEY`,
          `t5`.`bfuid_col_3299`
        FROM (
          SELECT
            `t0`.`C_CUSTKEY`
          FROM (
            SELECT
              `C_CUSTKEY`
            FROM `bigframes-dev.tpch.CUSTOMER` FOR SYSTEM_TIME AS OF CAST('2026-03-10T18:00:00' AS DATETIME)
          ) AS `t0`
        ) AS `t3`
        LEFT OUTER JOIN (
          SELECT
            `t1`.`O_ORDERKEY` AS `bfuid_col_3299`,
            `t1`.`O_CUSTKEY` AS `bfuid_col_3300`
          FROM (
            SELECT
              `O_ORDERKEY`,
              `O_CUSTKEY`,
              `O_COMMENT`
            FROM `bigframes-dev.tpch.ORDERS` FOR SYSTEM_TIME AS OF CAST('2026-03-10T18:00:00' AS DATETIME)
          ) AS `t1`
          WHERE
            NOT (
              regexp_contains(`t1`.`O_COMMENT`, 'special.*requests')
            )
        ) AS `t5`
          ON COALESCE(`t3`.`C_CUSTKEY`, 0) = COALESCE(`t5`.`bfuid_col_3300`, 0)
          AND COALESCE(`t3`.`C_CUSTKEY`, 1) = COALESCE(`t5`.`bfuid_col_3300`, 1)
      ) AS `t6`
      GROUP BY
        1
    ) AS `t7`
    WHERE
      (
        `t7`.`C_CUSTKEY`
      ) IS NOT NULL
  ) AS `t8`
  GROUP BY
    1
) AS `t9`
WHERE
  (
    `t9`.`bfuid_col_3309`
  ) IS NOT NULL) AS `t`
ORDER BY `CUSTDIST` DESC NULLS LAST ,`C_COUNT` DESC NULLS LAST