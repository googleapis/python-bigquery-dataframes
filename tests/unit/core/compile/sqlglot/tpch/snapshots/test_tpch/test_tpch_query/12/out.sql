SELECT `L_SHIPMODE`, `HIGH_LINE_COUNT`, `LOW_LINE_COUNT` FROM (SELECT
  `t8`.`bfuid_col_3291` AS `L_SHIPMODE`,
  `t8`.`bfuid_col_3295` AS `HIGH_LINE_COUNT`,
  `t8`.`bfuid_col_3296` AS `LOW_LINE_COUNT`
FROM (
  SELECT
    `t7`.`bfuid_col_3291`,
    COALESCE(SUM(`t7`.`bfuid_col_3293`), 0) AS `bfuid_col_3295`,
    COALESCE(SUM(`t7`.`bfuid_col_3294`), 0) AS `bfuid_col_3296`
  FROM (
    SELECT
      `t6`.`bfuid_col_3232` AS `bfuid_col_3291`,
      CAST(COALESCE(COALESCE(`t6`.`bfuid_col_3214` IN ('1-URGENT', '2-HIGH'), FALSE), FALSE) AS INT64) AS `bfuid_col_3293`,
      CAST(NOT (
        COALESCE(COALESCE(`t6`.`bfuid_col_3214` IN ('1-URGENT', '2-HIGH'), FALSE), FALSE)
      ) AS INT64) AS `bfuid_col_3294`
    FROM (
      SELECT
        `t4`.`O_ORDERPRIORITY` AS `bfuid_col_3214`,
        `t5`.`L_SHIPMODE` AS `bfuid_col_3232`,
        (
          (
            (
              COALESCE(COALESCE(`t5`.`L_SHIPMODE` IN ('MAIL', 'SHIP'), FALSE), FALSE)
              AND (
                `t5`.`L_COMMITDATE` < `t5`.`L_RECEIPTDATE`
              )
            )
            AND (
              `t5`.`L_SHIPDATE` < `t5`.`L_COMMITDATE`
            )
          )
          AND (
            `t5`.`L_RECEIPTDATE` >= DATE(1994, 1, 1)
          )
        )
        AND (
          `t5`.`L_RECEIPTDATE` < DATE(1995, 1, 1)
        ) AS `bfuid_col_3234`
      FROM (
        SELECT
          `t0`.`O_ORDERKEY`,
          `t0`.`O_ORDERPRIORITY`
        FROM (
          SELECT
            `O_ORDERKEY`,
            `O_ORDERPRIORITY`
          FROM `bigframes-dev.tpch.ORDERS` FOR SYSTEM_TIME AS OF CAST('2026-03-10T18:00:00' AS DATETIME)
        ) AS `t0`
      ) AS `t4`
      INNER JOIN (
        SELECT
          `t1`.`L_ORDERKEY`,
          `t1`.`L_SHIPDATE`,
          `t1`.`L_COMMITDATE`,
          `t1`.`L_RECEIPTDATE`,
          `t1`.`L_SHIPMODE`
        FROM (
          SELECT
            `L_ORDERKEY`,
            `L_SHIPDATE`,
            `L_COMMITDATE`,
            `L_RECEIPTDATE`,
            `L_SHIPMODE`
          FROM `bigframes-dev.tpch.LINEITEM` FOR SYSTEM_TIME AS OF CAST('2026-03-10T18:00:00' AS DATETIME)
        ) AS `t1`
      ) AS `t5`
        ON COALESCE(`t4`.`O_ORDERKEY`, 0) = COALESCE(`t5`.`L_ORDERKEY`, 0)
        AND COALESCE(`t4`.`O_ORDERKEY`, 1) = COALESCE(`t5`.`L_ORDERKEY`, 1)
    ) AS `t6`
    WHERE
      `t6`.`bfuid_col_3234`
  ) AS `t7`
  GROUP BY
    1
) AS `t8`
WHERE
  (
    `t8`.`bfuid_col_3291`
  ) IS NOT NULL) AS `t`
ORDER BY `L_SHIPMODE` ASC NULLS LAST