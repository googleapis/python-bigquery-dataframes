SELECT
  CASE
    WHEN `t3`.`col_1` = 0
    THEN `t9`.`bfuid_col_2247-sum`
    ELSE CAST(NULL AS FLOAT64)
  END AS `REVENUE`
FROM (
  SELECT
    *
  FROM (
    SELECT
      COALESCE(SUM(`t6`.`bfuid_col_2247`), 0) AS `bfuid_col_2247-sum`
    FROM (
      SELECT
        `t5`.`bfuid_col_2215` * `t5`.`bfuid_col_2216` AS `bfuid_col_2247`
      FROM (
        SELECT
          `t4`.`bfuid_col_2192` AS `bfuid_col_2214`,
          `t4`.`bfuid_col_2193` AS `bfuid_col_2215`,
          `t4`.`bfuid_col_2194` AS `bfuid_col_2216`,
          (
            `t4`.`bfuid_col_2194` >= 0.05
          ) AND (
            `t4`.`bfuid_col_2194` <= 0.07
          ) AS `bfuid_col_2226`
        FROM (
          SELECT
            `t1`.`L_QUANTITY` AS `bfuid_col_2192`,
            `t1`.`L_EXTENDEDPRICE` AS `bfuid_col_2193`,
            `t1`.`L_DISCOUNT` AS `bfuid_col_2194`,
            (
              `t1`.`L_SHIPDATE` >= DATE(1994, 1, 1)
            )
            AND (
              `t1`.`L_SHIPDATE` < DATE(1995, 1, 1)
            ) AS `bfuid_col_2204`
          FROM (
            SELECT
              `L_QUANTITY`,
              `L_EXTENDEDPRICE`,
              `L_DISCOUNT`,
              `L_SHIPDATE`
            FROM `bigframes-dev.tpch.LINEITEM` FOR SYSTEM_TIME AS OF CAST('2026-03-10T18:00:00' AS DATETIME)
          ) AS `t1`
        ) AS `t4`
        WHERE
          `t4`.`bfuid_col_2204`
      ) AS `t5`
      WHERE
        `t5`.`bfuid_col_2226` AND `t5`.`bfuid_col_2214` < 24
    ) AS `t6`
  ) AS `t7`
) AS `t9`
CROSS JOIN (
  SELECT
    *
  FROM (
    SELECT
      *
    FROM UNNEST(ARRAY<STRUCT<`col_1` INT64>>[STRUCT(0)]) AS `col_1`
  ) AS `t0`
) AS `t3`