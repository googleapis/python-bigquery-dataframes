SELECT `PS_PARTKEY`, `VALUE` FROM (SELECT
  `t30`.`bfuid_col_3183` AS `PS_PARTKEY`,
  `t30`.`bfuid_col_3184` AS `VALUE`
FROM (
  SELECT
    `t25`.`bfuid_col_3166` AS `bfuid_col_3183`,
    `t25`.`bfuid_col_3167` AS `bfuid_col_3184`,
    `t25`.`bfuid_col_3167` > `t29`.`bfuid_col_3179` AS `bfuid_col_3186`
  FROM (
    SELECT
      `t20`.`bfuid_col_3158` AS `bfuid_col_3166`,
      ROUND(`t20`.`bfuid_col_3164`, 2) AS `bfuid_col_3167`
    FROM (
      SELECT
        `t18`.`bfuid_col_3158`,
        COALESCE(SUM(`t18`.`bfuid_col_3163`), 0) AS `bfuid_col_3164`
      FROM (
        SELECT
          `t9`.`PS_PARTKEY` AS `bfuid_col_3158`,
          `t9`.`PS_SUPPLYCOST` * `t9`.`PS_AVAILQTY` AS `bfuid_col_3163`
        FROM (
          SELECT
            *
          FROM (
            SELECT
              `t12`.`S_SUPPKEY`
            FROM (
              SELECT
                `t2`.`N_NATIONKEY` AS `bfuid_col_3139`
              FROM (
                SELECT
                  `N_NATIONKEY`,
                  `N_NAME`
                FROM `bigframes-dev.tpch.NATION` FOR SYSTEM_TIME AS OF CAST('2026-03-10T18:00:00' AS DATETIME)
              ) AS `t2`
              WHERE
                `t2`.`N_NAME` = 'GERMANY'
            ) AS `t14`
            INNER JOIN (
              SELECT
                `t3`.`S_SUPPKEY`,
                `t3`.`S_NATIONKEY`
              FROM (
                SELECT
                  `S_SUPPKEY`,
                  `S_NATIONKEY`
                FROM `bigframes-dev.tpch.SUPPLIER` FOR SYSTEM_TIME AS OF CAST('2026-03-10T18:00:00' AS DATETIME)
              ) AS `t3`
            ) AS `t12`
              ON COALESCE(`t14`.`bfuid_col_3139`, 0) = COALESCE(`t12`.`S_NATIONKEY`, 0)
              AND COALESCE(`t14`.`bfuid_col_3139`, 1) = COALESCE(`t12`.`S_NATIONKEY`, 1)
          ) AS `t15`
        ) AS `t17`
        INNER JOIN (
          SELECT
            `t0`.`PS_PARTKEY`,
            `t0`.`PS_SUPPKEY`,
            `t0`.`PS_AVAILQTY`,
            `t0`.`PS_SUPPLYCOST`
          FROM (
            SELECT
              `PS_PARTKEY`,
              `PS_SUPPKEY`,
              `PS_AVAILQTY`,
              `PS_SUPPLYCOST`
            FROM `bigframes-dev.tpch.PARTSUPP` FOR SYSTEM_TIME AS OF CAST('2026-03-10T18:00:00' AS DATETIME)
          ) AS `t0`
        ) AS `t9`
          ON COALESCE(`t17`.`S_SUPPKEY`, 0) = COALESCE(`t9`.`PS_SUPPKEY`, 0)
          AND COALESCE(`t17`.`S_SUPPKEY`, 1) = COALESCE(`t9`.`PS_SUPPKEY`, 1)
      ) AS `t18`
      GROUP BY
        1
    ) AS `t20`
    WHERE
      (
        `t20`.`bfuid_col_3158`
      ) IS NOT NULL
  ) AS `t25`
  CROSS JOIN (
    SELECT
      `t27`.`bfuid_col_3178` * 0.0001 AS `bfuid_col_3179`
    FROM (
      SELECT
        `t26`.`bfuid_col_3175`,
        `t26`.`bfuid_col_3176`,
        ANY_VALUE(`t26`.`bfuid_col_3177`) AS `bfuid_col_3178`
      FROM (
        SELECT
          `t10`.`col_0` AS `bfuid_col_3175`,
          `t10`.`col_1` AS `bfuid_col_3176`,
          CASE
            WHEN `t24`.`bfuid_col_3173` = 0
            THEN CASE
              WHEN `t10`.`col_2` = 0
              THEN `t24`.`bfuid_col_3171`
              ELSE CAST(NULL AS FLOAT64)
            END
            ELSE CAST(NULL AS FLOAT64)
          END AS `bfuid_col_3177`
        FROM (
          SELECT
            *
          FROM (
            SELECT
              *
            FROM UNNEST(ARRAY<STRUCT<`col_0` FLOAT64, `col_1` INT64, `col_2` INT64>>[STRUCT(0.0, 0, 0)]) AS `col_0`
          ) AS `t1`
        ) AS `t10`
        CROSS JOIN (
          SELECT
            `t21`.`bfuid_col_3171`,
            0 AS `bfuid_col_3173`
          FROM (
            SELECT
              COALESCE(SUM(`t19`.`bfuid_col_3171`), 0) AS `bfuid_col_3171`
            FROM (
              SELECT
                `t13`.`PS_SUPPLYCOST` * `t13`.`PS_AVAILQTY` AS `bfuid_col_3171`
              FROM (
                SELECT
                  *
                FROM (
                  SELECT
                    `t12`.`S_SUPPKEY`
                  FROM (
                    SELECT
                      `t2`.`N_NATIONKEY` AS `bfuid_col_3139`
                    FROM (
                      SELECT
                        `N_NATIONKEY`,
                        `N_NAME`
                      FROM `bigframes-dev.tpch.NATION` FOR SYSTEM_TIME AS OF CAST('2026-03-10T18:00:00' AS DATETIME)
                    ) AS `t2`
                    WHERE
                      `t2`.`N_NAME` = 'GERMANY'
                  ) AS `t14`
                  INNER JOIN (
                    SELECT
                      `t3`.`S_SUPPKEY`,
                      `t3`.`S_NATIONKEY`
                    FROM (
                      SELECT
                        `S_SUPPKEY`,
                        `S_NATIONKEY`
                      FROM `bigframes-dev.tpch.SUPPLIER` FOR SYSTEM_TIME AS OF CAST('2026-03-10T18:00:00' AS DATETIME)
                    ) AS `t3`
                  ) AS `t12`
                    ON COALESCE(`t14`.`bfuid_col_3139`, 0) = COALESCE(`t12`.`S_NATIONKEY`, 0)
                    AND COALESCE(`t14`.`bfuid_col_3139`, 1) = COALESCE(`t12`.`S_NATIONKEY`, 1)
                ) AS `t15`
              ) AS `t17`
              INNER JOIN (
                SELECT
                  `t4`.`PS_SUPPKEY`,
                  `t4`.`PS_AVAILQTY`,
                  `t4`.`PS_SUPPLYCOST`
                FROM (
                  SELECT
                    `PS_SUPPKEY`,
                    `PS_AVAILQTY`,
                    `PS_SUPPLYCOST`
                  FROM `bigframes-dev.tpch.PARTSUPP` FOR SYSTEM_TIME AS OF CAST('2026-03-10T18:00:00' AS DATETIME)
                ) AS `t4`
              ) AS `t13`
                ON COALESCE(`t17`.`S_SUPPKEY`, 0) = COALESCE(`t13`.`PS_SUPPKEY`, 0)
                AND COALESCE(`t17`.`S_SUPPKEY`, 1) = COALESCE(`t13`.`PS_SUPPKEY`, 1)
            ) AS `t19`
          ) AS `t21`
        ) AS `t24`
      ) AS `t26`
      GROUP BY
        1,
        2
    ) AS `t27`
    WHERE
      (
        `t27`.`bfuid_col_3175`
      ) IS NOT NULL
      AND (
        `t27`.`bfuid_col_3176`
      ) IS NOT NULL
  ) AS `t29`
) AS `t30`
WHERE
  `t30`.`bfuid_col_3186`) AS `t`
ORDER BY `VALUE` DESC NULLS LAST