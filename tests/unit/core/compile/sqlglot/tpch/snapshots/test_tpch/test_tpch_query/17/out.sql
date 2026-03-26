SELECT `AVG_YEARLY` FROM (SELECT
  ROUND(ieee_divide(`t26`.`bfuid_col_3612`, 7.0), 2) AS `AVG_YEARLY`,
  `t26`.`bfuid_col_3610` AS `bfuid_col_3620`,
  `t26`.`bfuid_col_3609` AS `bfuid_col_3621`
FROM (
  SELECT
    `t25`.`bfuid_col_3609`,
    `t25`.`bfuid_col_3610`,
    ANY_VALUE(`t25`.`bfuid_col_3611`) AS `bfuid_col_3612`
  FROM (
    SELECT
      `t7`.`col_0` AS `bfuid_col_3609`,
      `t7`.`col_1` AS `bfuid_col_3610`,
      CASE
        WHEN `t24`.`bfuid_col_3607` = 0
        THEN CASE
          WHEN `t7`.`col_2` = 0
          THEN `t24`.`bfuid_col_3585`
          ELSE CAST(NULL AS FLOAT64)
        END
        ELSE CAST(NULL AS FLOAT64)
      END AS `bfuid_col_3611`
    FROM (
      SELECT
        *
      FROM (
        SELECT
          *
        FROM UNNEST(ARRAY<STRUCT<`col_0` STRING, `col_1` INT64, `col_2` INT64>>[STRUCT('L_EXTENDEDPRICE', 0, 0)]) AS `col_0`
      ) AS `t0`
    ) AS `t7`
    CROSS JOIN (
      SELECT
        `t22`.`bfuid_col_3585`,
        0 AS `bfuid_col_3607`
      FROM (
        SELECT
          COALESCE(SUM(`t21`.`bfuid_col_3585`), 0) AS `bfuid_col_3585`
        FROM (
          SELECT
            `t20`.`bfuid_col_3585`
          FROM (
            SELECT
              `t17`.`L_EXTENDEDPRICE` AS `bfuid_col_3585`,
              `t17`.`L_QUANTITY` < `t19`.`bfuid_col_3573` AS `bfuid_col_3605`
            FROM (
              SELECT
                `t16`.`bfuid_col_3560` AS `bfuid_col_3572`,
                `t16`.`bfuid_col_3570` * 0.2 AS `bfuid_col_3573`
              FROM (
                SELECT
                  `t13`.`bfuid_col_3560`,
                  AVG(`t13`.`L_QUANTITY`) AS `bfuid_col_3570`
                FROM (
                  SELECT
                    `t8`.`L_QUANTITY`,
                    `t12`.`bfuid_col_3560`
                  FROM (
                    SELECT
                      `t1`.`L_PARTKEY`,
                      `t1`.`L_QUANTITY`
                    FROM (
                      SELECT
                        `L_PARTKEY`,
                        `L_QUANTITY`
                      FROM `bigframes-dev.tpch.LINEITEM` FOR SYSTEM_TIME AS OF CAST('2026-03-10T18:00:00' AS DATETIME)
                    ) AS `t1`
                  ) AS `t8`
                  RIGHT OUTER JOIN (
                    SELECT
                      `t10`.`bfuid_col_3560`
                    FROM (
                      SELECT
                        `t3`.`P_PARTKEY` AS `bfuid_col_3560`,
                        (
                          `t3`.`P_BRAND` = 'Brand#23'
                        ) AND (
                          `t3`.`P_CONTAINER` = 'MED BOX'
                        ) AS `bfuid_col_3569`
                      FROM (
                        SELECT
                          `P_PARTKEY`,
                          `P_BRAND`,
                          `P_CONTAINER`
                        FROM `bigframes-dev.tpch.PART` FOR SYSTEM_TIME AS OF CAST('2026-03-10T18:00:00' AS DATETIME)
                      ) AS `t3`
                    ) AS `t10`
                    WHERE
                      `t10`.`bfuid_col_3569`
                  ) AS `t12`
                    ON COALESCE(`t8`.`L_PARTKEY`, 0) = COALESCE(`t12`.`bfuid_col_3560`, 0)
                    AND COALESCE(`t8`.`L_PARTKEY`, 1) = COALESCE(`t12`.`bfuid_col_3560`, 1)
                ) AS `t13`
                GROUP BY
                  1
              ) AS `t16`
              WHERE
                (
                  `t16`.`bfuid_col_3560`
                ) IS NOT NULL
            ) AS `t19`
            INNER JOIN (
              SELECT
                *
              FROM (
                SELECT
                  `t9`.`L_QUANTITY`,
                  `t9`.`L_EXTENDEDPRICE`,
                  `t12`.`bfuid_col_3560`
                FROM (
                  SELECT
                    `t2`.`L_PARTKEY`,
                    `t2`.`L_QUANTITY`,
                    `t2`.`L_EXTENDEDPRICE`
                  FROM (
                    SELECT
                      `L_PARTKEY`,
                      `L_QUANTITY`,
                      `L_EXTENDEDPRICE`
                    FROM `bigframes-dev.tpch.LINEITEM` FOR SYSTEM_TIME AS OF CAST('2026-03-10T18:00:00' AS DATETIME)
                  ) AS `t2`
                ) AS `t9`
                RIGHT OUTER JOIN (
                  SELECT
                    `t10`.`bfuid_col_3560`
                  FROM (
                    SELECT
                      `t3`.`P_PARTKEY` AS `bfuid_col_3560`,
                      (
                        `t3`.`P_BRAND` = 'Brand#23'
                      ) AND (
                        `t3`.`P_CONTAINER` = 'MED BOX'
                      ) AS `bfuid_col_3569`
                    FROM (
                      SELECT
                        `P_PARTKEY`,
                        `P_BRAND`,
                        `P_CONTAINER`
                      FROM `bigframes-dev.tpch.PART` FOR SYSTEM_TIME AS OF CAST('2026-03-10T18:00:00' AS DATETIME)
                    ) AS `t3`
                  ) AS `t10`
                  WHERE
                    `t10`.`bfuid_col_3569`
                ) AS `t12`
                  ON COALESCE(`t9`.`L_PARTKEY`, 0) = COALESCE(`t12`.`bfuid_col_3560`, 0)
                  AND COALESCE(`t9`.`L_PARTKEY`, 1) = COALESCE(`t12`.`bfuid_col_3560`, 1)
              ) AS `t14`
            ) AS `t17`
              ON `t19`.`bfuid_col_3572` = `t17`.`bfuid_col_3560`
          ) AS `t20`
          WHERE
            `t20`.`bfuid_col_3605`
        ) AS `t21`
      ) AS `t22`
    ) AS `t24`
  ) AS `t25`
  GROUP BY
    1,
    2
) AS `t26`
WHERE
  (
    `t26`.`bfuid_col_3609`
  ) IS NOT NULL
  AND (
    `t26`.`bfuid_col_3610`
  ) IS NOT NULL) AS `t`
ORDER BY `bfuid_col_3620` ASC NULLS LAST ,`bfuid_col_3621` ASC NULLS LAST