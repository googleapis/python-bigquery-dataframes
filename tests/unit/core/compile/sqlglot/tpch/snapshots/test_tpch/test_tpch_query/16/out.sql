SELECT `P_BRAND`, `P_TYPE`, `P_SIZE`, `SUPPLIER_CNT` FROM (SELECT
  `t15`.`bfuid_col_3525` AS `P_BRAND`,
  `t15`.`bfuid_col_3526` AS `P_TYPE`,
  `t15`.`bfuid_col_3527` AS `P_SIZE`,
  `t15`.`bfuid_col_3554` AS `SUPPLIER_CNT`
FROM (
  SELECT
    `t14`.`bfuid_col_3525`,
    `t14`.`bfuid_col_3526`,
    `t14`.`bfuid_col_3527`,
    COUNT(DISTINCT `t14`.`bfuid_col_3532`) AS `bfuid_col_3554`
  FROM (
    SELECT
      `t13`.`bfuid_col_3525`,
      `t13`.`bfuid_col_3526`,
      `t13`.`bfuid_col_3527`,
      `t13`.`bfuid_col_3532`
    FROM (
      SELECT
        `t11`.`bfuid_col_3525`,
        `t11`.`bfuid_col_3526`,
        `t11`.`bfuid_col_3527`,
        `t11`.`bfuid_col_3532`,
        EXISTS(
          SELECT
            1
          FROM (
            SELECT
              `t7`.`bfuid_col_3479`
            FROM (
              SELECT
                `t0`.`S_SUPPKEY` AS `bfuid_col_3479`
              FROM (
                SELECT
                  `S_SUPPKEY`,
                  `S_COMMENT`
                FROM `bigframes-dev.tpch.SUPPLIER` FOR SYSTEM_TIME AS OF CAST('2026-03-10T18:00:00' AS DATETIME)
              ) AS `t0`
              WHERE
                NOT (
                  regexp_contains(`t0`.`S_COMMENT`, 'Customer.*Complaints')
                )
            ) AS `t7`
            GROUP BY
              1
          ) AS `t8`
          WHERE
            (
              COALESCE(`t11`.`bfuid_col_3532`, 0) = COALESCE(`t8`.`bfuid_col_3479`, 0)
            )
            AND (
              COALESCE(`t11`.`bfuid_col_3532`, 1) = COALESCE(`t8`.`bfuid_col_3479`, 1)
            )
        ) AS `bfuid_col_3537`
      FROM (
        SELECT
          `t10`.`bfuid_col_3525`,
          `t10`.`bfuid_col_3526`,
          `t10`.`bfuid_col_3527`,
          `t10`.`bfuid_col_3532`
        FROM (
          SELECT
            `t9`.`bfuid_col_3491` AS `bfuid_col_3525`,
            `t9`.`bfuid_col_3492` AS `bfuid_col_3526`,
            `t9`.`bfuid_col_3493` AS `bfuid_col_3527`,
            `t9`.`bfuid_col_3498` AS `bfuid_col_3532`,
            COALESCE(COALESCE(`t9`.`bfuid_col_3493` IN (49, 14, 23, 45, 19, 3, 36, 9), FALSE), FALSE) AS `bfuid_col_3536`
          FROM (
            SELECT
              `t5`.`P_BRAND` AS `bfuid_col_3491`,
              `t5`.`P_TYPE` AS `bfuid_col_3492`,
              `t5`.`P_SIZE` AS `bfuid_col_3493`,
              `t6`.`PS_SUPPKEY` AS `bfuid_col_3498`,
              `t5`.`P_BRAND` <> 'Brand#45' AS `bfuid_col_3502`
            FROM (
              SELECT
                `t1`.`P_PARTKEY`,
                `t1`.`P_BRAND`,
                `t1`.`P_TYPE`,
                `t1`.`P_SIZE`
              FROM (
                SELECT
                  `P_PARTKEY`,
                  `P_BRAND`,
                  `P_TYPE`,
                  `P_SIZE`
                FROM `bigframes-dev.tpch.PART` FOR SYSTEM_TIME AS OF CAST('2026-03-10T18:00:00' AS DATETIME)
              ) AS `t1`
            ) AS `t5`
            INNER JOIN (
              SELECT
                `t2`.`PS_PARTKEY`,
                `t2`.`PS_SUPPKEY`
              FROM (
                SELECT
                  `PS_PARTKEY`,
                  `PS_SUPPKEY`
                FROM `bigframes-dev.tpch.PARTSUPP` FOR SYSTEM_TIME AS OF CAST('2026-03-10T18:00:00' AS DATETIME)
              ) AS `t2`
            ) AS `t6`
              ON COALESCE(`t5`.`P_PARTKEY`, 0) = COALESCE(`t6`.`PS_PARTKEY`, 0)
              AND COALESCE(`t5`.`P_PARTKEY`, 1) = COALESCE(`t6`.`PS_PARTKEY`, 1)
          ) AS `t9`
          WHERE
            `t9`.`bfuid_col_3502`
            AND NOT (
              regexp_contains(`t9`.`bfuid_col_3492`, 'MEDIUM POLISHED')
            )
        ) AS `t10`
        WHERE
          `t10`.`bfuid_col_3536`
      ) AS `t11`
    ) AS `t13`
    WHERE
      `t13`.`bfuid_col_3537`
  ) AS `t14`
  GROUP BY
    1,
    2,
    3
) AS `t15`
WHERE
  (
    `t15`.`bfuid_col_3525`
  ) IS NOT NULL
  AND (
    `t15`.`bfuid_col_3526`
  ) IS NOT NULL
  AND (
    `t15`.`bfuid_col_3527`
  ) IS NOT NULL) AS `t`
ORDER BY `SUPPLIER_CNT` DESC NULLS LAST ,`P_BRAND` ASC NULLS LAST ,`P_TYPE` ASC NULLS LAST ,`P_SIZE` ASC NULLS LAST