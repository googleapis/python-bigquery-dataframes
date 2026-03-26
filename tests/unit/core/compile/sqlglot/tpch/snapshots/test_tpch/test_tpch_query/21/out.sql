SELECT `S_NAME`, `NUMWAIT` FROM (SELECT
  `t38`.`bfuid_col_3886` AS `S_NAME`,
  `t38`.`bfuid_col_3906` AS `NUMWAIT`
FROM (
  SELECT
    `t37`.`bfuid_col_3886`,
    COUNT(1) AS `bfuid_col_3906`
  FROM (
    SELECT
      `t36`.`bfuid_col_3886`
    FROM (
      SELECT
        `t35`.`S_NAME` AS `bfuid_col_3886`,
        (
          (
            `t35`.`bfuid_col_3856` = 1
          ) AND (
            `t35`.`N_NAME` = 'SAUDI ARABIA'
          )
        )
        AND (
          `t10`.`O_ORDERSTATUS` = 'F'
        ) AS `bfuid_col_3905`
      FROM (
        SELECT
          *
        FROM (
          SELECT
            `t32`.`bfuid_col_3833`,
            `t32`.`bfuid_col_3856`,
            `t32`.`S_NAME`,
            `t11`.`N_NAME`
          FROM (
            SELECT
              *
            FROM (
              SELECT
                `t29`.`bfuid_col_3833`,
                `t29`.`bfuid_col_3856`,
                `t12`.`S_NAME`,
                `t12`.`S_NATIONKEY`
              FROM (
                SELECT
                  *
                FROM (
                  SELECT
                    `t26`.`bfuid_col_3833`,
                    `t26`.`bfuid_col_3856`,
                    `t24`.`bfuid_col_3841`
                  FROM (
                    SELECT
                      *
                    FROM (
                      SELECT
                        `t20`.`bfuid_col_3833`,
                        COUNT(1) AS `bfuid_col_3856`
                      FROM (
                        SELECT
                          `t19`.`bfuid_col_3833`
                        FROM (
                          SELECT
                            `t17`.`L_ORDERKEY` AS `bfuid_col_3833`
                          FROM (
                            SELECT
                              `t9`.`L_ORDERKEY`,
                              COUNT(1) AS `bfuid_col_3831`
                            FROM (
                              SELECT
                                `t5`.`L_ORDERKEY`
                              FROM (
                                SELECT
                                  `L_ORDERKEY`
                                FROM `bigframes-dev.tpch.LINEITEM` FOR SYSTEM_TIME AS OF CAST('2026-03-10T18:00:00' AS DATETIME)
                              ) AS `t5`
                            ) AS `t9`
                            GROUP BY
                              1
                          ) AS `t17`
                          WHERE
                            (
                              `t17`.`L_ORDERKEY`
                            ) IS NOT NULL AND `t17`.`bfuid_col_3831` > 1
                        ) AS `t19`
                        INNER JOIN (
                          SELECT
                            `t3`.`L_ORDERKEY` AS `bfuid_col_3839`
                          FROM (
                            SELECT
                              `L_ORDERKEY`,
                              `L_COMMITDATE`,
                              `L_RECEIPTDATE`
                            FROM `bigframes-dev.tpch.LINEITEM` FOR SYSTEM_TIME AS OF CAST('2026-03-10T18:00:00' AS DATETIME)
                          ) AS `t3`
                          WHERE
                            `t3`.`L_RECEIPTDATE` > `t3`.`L_COMMITDATE`
                        ) AS `t15`
                          ON `t19`.`bfuid_col_3833` = `t15`.`bfuid_col_3839`
                      ) AS `t20`
                      GROUP BY
                        1
                    ) AS `t22`
                  ) AS `t26`
                  INNER JOIN (
                    SELECT
                      *
                    FROM (
                      SELECT
                        `t19`.`bfuid_col_3833` AS `bfuid_col_3857`,
                        `t16`.`bfuid_col_3841`
                      FROM (
                        SELECT
                          `t17`.`L_ORDERKEY` AS `bfuid_col_3833`
                        FROM (
                          SELECT
                            `t9`.`L_ORDERKEY`,
                            COUNT(1) AS `bfuid_col_3831`
                          FROM (
                            SELECT
                              `t5`.`L_ORDERKEY`
                            FROM (
                              SELECT
                                `L_ORDERKEY`
                              FROM `bigframes-dev.tpch.LINEITEM` FOR SYSTEM_TIME AS OF CAST('2026-03-10T18:00:00' AS DATETIME)
                            ) AS `t5`
                          ) AS `t9`
                          GROUP BY
                            1
                        ) AS `t17`
                        WHERE
                          (
                            `t17`.`L_ORDERKEY`
                          ) IS NOT NULL AND `t17`.`bfuid_col_3831` > 1
                      ) AS `t19`
                      INNER JOIN (
                        SELECT
                          `t4`.`L_ORDERKEY` AS `bfuid_col_3839`,
                          `t4`.`L_SUPPKEY` AS `bfuid_col_3841`
                        FROM (
                          SELECT
                            `L_ORDERKEY`,
                            `L_SUPPKEY`,
                            `L_COMMITDATE`,
                            `L_RECEIPTDATE`
                          FROM `bigframes-dev.tpch.LINEITEM` FOR SYSTEM_TIME AS OF CAST('2026-03-10T18:00:00' AS DATETIME)
                        ) AS `t4`
                        WHERE
                          `t4`.`L_RECEIPTDATE` > `t4`.`L_COMMITDATE`
                      ) AS `t16`
                        ON `t19`.`bfuid_col_3833` = `t16`.`bfuid_col_3839`
                    ) AS `t21`
                  ) AS `t24`
                    ON `t26`.`bfuid_col_3833` = `t24`.`bfuid_col_3857`
                ) AS `t27`
              ) AS `t29`
              INNER JOIN (
                SELECT
                  `t2`.`S_SUPPKEY`,
                  `t2`.`S_NAME`,
                  `t2`.`S_NATIONKEY`
                FROM (
                  SELECT
                    `S_SUPPKEY`,
                    `S_NAME`,
                    `S_NATIONKEY`
                  FROM `bigframes-dev.tpch.SUPPLIER` FOR SYSTEM_TIME AS OF CAST('2026-03-10T18:00:00' AS DATETIME)
                ) AS `t2`
              ) AS `t12`
                ON COALESCE(`t29`.`bfuid_col_3841`, 0) = COALESCE(`t12`.`S_SUPPKEY`, 0)
                AND COALESCE(`t29`.`bfuid_col_3841`, 1) = COALESCE(`t12`.`S_SUPPKEY`, 1)
            ) AS `t30`
          ) AS `t32`
          INNER JOIN (
            SELECT
              `t1`.`N_NATIONKEY`,
              `t1`.`N_NAME`
            FROM (
              SELECT
                `N_NATIONKEY`,
                `N_NAME`
              FROM `bigframes-dev.tpch.NATION` FOR SYSTEM_TIME AS OF CAST('2026-03-10T18:00:00' AS DATETIME)
            ) AS `t1`
          ) AS `t11`
            ON COALESCE(`t32`.`S_NATIONKEY`, 0) = COALESCE(`t11`.`N_NATIONKEY`, 0)
            AND COALESCE(`t32`.`S_NATIONKEY`, 1) = COALESCE(`t11`.`N_NATIONKEY`, 1)
        ) AS `t33`
      ) AS `t35`
      INNER JOIN (
        SELECT
          `t0`.`O_ORDERKEY`,
          `t0`.`O_ORDERSTATUS`
        FROM (
          SELECT
            `O_ORDERKEY`,
            `O_ORDERSTATUS`
          FROM `bigframes-dev.tpch.ORDERS` FOR SYSTEM_TIME AS OF CAST('2026-03-10T18:00:00' AS DATETIME)
        ) AS `t0`
      ) AS `t10`
        ON `t35`.`bfuid_col_3833` = `t10`.`O_ORDERKEY`
    ) AS `t36`
    WHERE
      `t36`.`bfuid_col_3905`
  ) AS `t37`
  GROUP BY
    1
) AS `t38`
WHERE
  (
    `t38`.`bfuid_col_3886`
  ) IS NOT NULL) AS `t`
ORDER BY `NUMWAIT` DESC NULLS LAST ,`S_NAME` ASC NULLS LAST
LIMIT 100