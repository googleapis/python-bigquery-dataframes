SELECT `CNTRYCODE`, `NUMCUST`, `TOTACCTBAL` FROM (SELECT
  `t25`.`bfuid_col_3962` AS `CNTRYCODE`,
  `t25`.`bfuid_col_3980` AS `NUMCUST`,
  `t25`.`bfuid_col_3981` AS `TOTACCTBAL`
FROM (
  SELECT
    `t24`.`bfuid_col_3962`,
    COUNT(`t24`.`bfuid_col_3954`) AS `bfuid_col_3980`,
    COALESCE(SUM(`t24`.`bfuid_col_3959`), 0) AS `bfuid_col_3981`
  FROM (
    SELECT
      `t23`.`bfuid_col_3954`,
      `t23`.`bfuid_col_3959`,
      `t23`.`bfuid_col_3962`
    FROM (
      SELECT
        `t21`.`bfuid_col_3954`,
        `t21`.`bfuid_col_3959`,
        `t21`.`bfuid_col_3962`,
        NOT (
          EXISTS(
            SELECT
              1
            FROM (
              SELECT
                `t4`.`O_CUSTKEY`
              FROM (
                SELECT
                  `t1`.`O_CUSTKEY`
                FROM (
                  SELECT
                    `O_CUSTKEY`
                  FROM `bigframes-dev.tpch.ORDERS` FOR SYSTEM_TIME AS OF CAST('2026-03-10T18:00:00' AS DATETIME)
                ) AS `t1`
              ) AS `t4`
              GROUP BY
                1
            ) AS `t7`
            WHERE
              (
                COALESCE(`t21`.`bfuid_col_3954`, 0) = COALESCE(`t7`.`O_CUSTKEY`, 0)
              )
              AND (
                COALESCE(`t21`.`bfuid_col_3954`, 1) = COALESCE(`t7`.`O_CUSTKEY`, 1)
              )
          )
        ) AS `bfuid_col_3966`
      FROM (
        SELECT
          `t20`.`bfuid_col_3954`,
          `t20`.`bfuid_col_3959`,
          `t20`.`bfuid_col_3962`
        FROM (
          SELECT
            `t11`.`bfuid_col_3919` AS `bfuid_col_3954`,
            `t11`.`bfuid_col_3924` AS `bfuid_col_3959`,
            `t11`.`bfuid_col_3927` AS `bfuid_col_3962`,
            `t11`.`bfuid_col_3924` > `t19`.`bfuid_col_3946` AS `bfuid_col_3964`
          FROM (
            SELECT
              `t8`.`bfuid_col_3919`,
              `t8`.`bfuid_col_3924`,
              `t8`.`bfuid_col_3927`
            FROM (
              SELECT
                `t0`.`C_CUSTKEY` AS `bfuid_col_3919`,
                `t0`.`C_ACCTBAL` AS `bfuid_col_3924`,
                SUBSTRING(
                  `t0`.`C_PHONE`,
                  IF(
                    (
                      IF(0 >= 0, 0, GREATEST(0, LENGTH(`t0`.`C_PHONE`) + 0)) + 1
                    ) >= 1,
                    IF(0 >= 0, 0, GREATEST(0, LENGTH(`t0`.`C_PHONE`) + 0)) + 1,
                    IF(0 >= 0, 0, GREATEST(0, LENGTH(`t0`.`C_PHONE`) + 0)) + 1 + LENGTH(`t0`.`C_PHONE`)
                  ),
                  GREATEST(
                    0,
                    IF(2 >= 0, 2, GREATEST(0, LENGTH(`t0`.`C_PHONE`) + 2)) - IF(0 >= 0, 0, GREATEST(0, LENGTH(`t0`.`C_PHONE`) + 0))
                  )
                ) AS `bfuid_col_3927`,
                COALESCE(
                  COALESCE(
                    SUBSTRING(
                      `t0`.`C_PHONE`,
                      IF(
                        (
                          IF(0 >= 0, 0, GREATEST(0, LENGTH(`t0`.`C_PHONE`) + 0)) + 1
                        ) >= 1,
                        IF(0 >= 0, 0, GREATEST(0, LENGTH(`t0`.`C_PHONE`) + 0)) + 1,
                        IF(0 >= 0, 0, GREATEST(0, LENGTH(`t0`.`C_PHONE`) + 0)) + 1 + LENGTH(`t0`.`C_PHONE`)
                      ),
                      GREATEST(
                        0,
                        IF(2 >= 0, 2, GREATEST(0, LENGTH(`t0`.`C_PHONE`) + 2)) - IF(0 >= 0, 0, GREATEST(0, LENGTH(`t0`.`C_PHONE`) + 0))
                      )
                    ) IN ('13', '31', '23', '29', '30', '18', '17'),
                    FALSE
                  ),
                  FALSE
                ) AS `bfuid_col_3928`
              FROM (
                SELECT
                  `C_CUSTKEY`,
                  `C_PHONE`,
                  `C_ACCTBAL`
                FROM `bigframes-dev.tpch.CUSTOMER` FOR SYSTEM_TIME AS OF CAST('2026-03-10T18:00:00' AS DATETIME)
              ) AS `t0`
            ) AS `t8`
            WHERE
              `t8`.`bfuid_col_3928`
          ) AS `t11`
          CROSS JOIN (
            SELECT
              `t17`.`bfuid_col_3946`
            FROM (
              SELECT
                `t16`.`bfuid_col_3943`,
                `t16`.`bfuid_col_3944`,
                ANY_VALUE(`t16`.`bfuid_col_3945`) AS `bfuid_col_3946`
              FROM (
                SELECT
                  `t6`.`col_0` AS `bfuid_col_3943`,
                  `t6`.`col_1` AS `bfuid_col_3944`,
                  CASE
                    WHEN `t15`.`bfuid_col_3941` = 0
                    THEN CASE
                      WHEN `t6`.`col_2` = 0
                      THEN `t15`.`bfuid_col_3935`
                      ELSE CAST(NULL AS FLOAT64)
                    END
                    ELSE CAST(NULL AS FLOAT64)
                  END AS `bfuid_col_3945`
                FROM (
                  SELECT
                    *
                  FROM (
                    SELECT
                      *
                    FROM UNNEST(ARRAY<STRUCT<`col_0` STRING, `col_1` INT64, `col_2` INT64>>[STRUCT('C_ACCTBAL', 0, 0)]) AS `col_0`
                  ) AS `t2`
                ) AS `t6`
                CROSS JOIN (
                  SELECT
                    `t13`.`bfuid_col_3935`,
                    0 AS `bfuid_col_3941`
                  FROM (
                    SELECT
                      AVG(`t12`.`bfuid_col_3935`) AS `bfuid_col_3935`
                    FROM (
                      SELECT
                        `t9`.`bfuid_col_3924` AS `bfuid_col_3935`
                      FROM (
                        SELECT
                          `t3`.`C_ACCTBAL` AS `bfuid_col_3924`,
                          COALESCE(
                            COALESCE(
                              SUBSTRING(
                                `t3`.`C_PHONE`,
                                IF(
                                  (
                                    IF(0 >= 0, 0, GREATEST(0, LENGTH(`t3`.`C_PHONE`) + 0)) + 1
                                  ) >= 1,
                                  IF(0 >= 0, 0, GREATEST(0, LENGTH(`t3`.`C_PHONE`) + 0)) + 1,
                                  IF(0 >= 0, 0, GREATEST(0, LENGTH(`t3`.`C_PHONE`) + 0)) + 1 + LENGTH(`t3`.`C_PHONE`)
                                ),
                                GREATEST(
                                  0,
                                  IF(2 >= 0, 2, GREATEST(0, LENGTH(`t3`.`C_PHONE`) + 2)) - IF(0 >= 0, 0, GREATEST(0, LENGTH(`t3`.`C_PHONE`) + 0))
                                )
                              ) IN ('13', '31', '23', '29', '30', '18', '17'),
                              FALSE
                            ),
                            FALSE
                          ) AS `bfuid_col_3928`
                        FROM (
                          SELECT
                            `C_PHONE`,
                            `C_ACCTBAL`
                          FROM `bigframes-dev.tpch.CUSTOMER` FOR SYSTEM_TIME AS OF CAST('2026-03-10T18:00:00' AS DATETIME)
                        ) AS `t3`
                      ) AS `t9`
                      WHERE
                        `t9`.`bfuid_col_3928` AND `t9`.`bfuid_col_3924` > 0.0
                    ) AS `t12`
                  ) AS `t13`
                ) AS `t15`
              ) AS `t16`
              GROUP BY
                1,
                2
            ) AS `t17`
            WHERE
              (
                `t17`.`bfuid_col_3943`
              ) IS NOT NULL
              AND (
                `t17`.`bfuid_col_3944`
              ) IS NOT NULL
          ) AS `t19`
        ) AS `t20`
        WHERE
          `t20`.`bfuid_col_3964`
      ) AS `t21`
    ) AS `t23`
    WHERE
      `t23`.`bfuid_col_3966`
  ) AS `t24`
  GROUP BY
    1
) AS `t25`
WHERE
  (
    `t25`.`bfuid_col_3962`
  ) IS NOT NULL) AS `t`
ORDER BY `CNTRYCODE` ASC NULLS LAST