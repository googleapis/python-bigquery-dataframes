SELECT
  CASE
    WHEN `t6`.`col_1` = 0
    THEN `t13`.`bfuid_col_3750-sum`
    ELSE CAST(NULL AS FLOAT64)
  END AS `REVENUE`
FROM (
  SELECT
    *
  FROM (
    SELECT
      COALESCE(SUM(`t10`.`bfuid_col_3750`), 0) AS `bfuid_col_3750-sum`
    FROM (
      SELECT
        `t9`.`bfuid_col_3735` * (
          1 - `t9`.`bfuid_col_3736`
        ) AS `bfuid_col_3750`
      FROM (
        SELECT
          `t8`.`L_EXTENDEDPRICE` AS `bfuid_col_3735`,
          `t8`.`L_DISCOUNT` AS `bfuid_col_3736`,
          (
            COALESCE(COALESCE(`t8`.`L_SHIPMODE` IN ('AIR', 'AIR REG'), FALSE), FALSE)
            AND (
              `t8`.`L_SHIPINSTRUCT` = 'DELIVER IN PERSON'
            )
          )
          AND (
            (
              (
                (
                  (
                    (
                      `t7`.`P_BRAND` = 'Brand#12'
                    )
                    AND COALESCE(
                      COALESCE(`t7`.`P_CONTAINER` IN ('SM CASE', 'SM BOX', 'SM PACK', 'SM PKG'), FALSE),
                      FALSE
                    )
                  )
                  AND (
                    (
                      `t8`.`L_QUANTITY` >= 1
                    ) AND (
                      `t8`.`L_QUANTITY` <= 11
                    )
                  )
                )
                AND (
                  (
                    `t7`.`P_SIZE` >= 1
                  ) AND (
                    `t7`.`P_SIZE` <= 5
                  )
                )
              )
              OR (
                (
                  (
                    (
                      `t7`.`P_BRAND` = 'Brand#23'
                    )
                    AND COALESCE(
                      COALESCE(`t7`.`P_CONTAINER` IN ('MED BAG', 'MED BOX', 'MED PKG', 'MED PACK'), FALSE),
                      FALSE
                    )
                  )
                  AND (
                    (
                      `t8`.`L_QUANTITY` >= 10
                    ) AND (
                      `t8`.`L_QUANTITY` <= 20
                    )
                  )
                )
                AND (
                  (
                    `t7`.`P_SIZE` >= 1
                  ) AND (
                    `t7`.`P_SIZE` <= 10
                  )
                )
              )
            )
            OR (
              (
                (
                  (
                    `t7`.`P_BRAND` = 'Brand#34'
                  )
                  AND COALESCE(
                    COALESCE(`t7`.`P_CONTAINER` IN ('LG CASE', 'LG BOX', 'LG PACK', 'LG PKG'), FALSE),
                    FALSE
                  )
                )
                AND (
                  (
                    `t8`.`L_QUANTITY` >= 20
                  ) AND (
                    `t8`.`L_QUANTITY` <= 30
                  )
                )
              )
              AND (
                (
                  `t7`.`P_SIZE` >= 1
                ) AND (
                  `t7`.`P_SIZE` <= 15
                )
              )
            )
          ) AS `bfuid_col_3746`
        FROM (
          SELECT
            `t1`.`P_PARTKEY`,
            `t1`.`P_BRAND`,
            `t1`.`P_SIZE`,
            `t1`.`P_CONTAINER`
          FROM (
            SELECT
              `P_PARTKEY`,
              `P_BRAND`,
              `P_SIZE`,
              `P_CONTAINER`
            FROM `bigframes-dev.tpch.PART` FOR SYSTEM_TIME AS OF CAST('2026-03-10T18:00:00' AS DATETIME)
          ) AS `t1`
        ) AS `t7`
        INNER JOIN (
          SELECT
            `t2`.`L_PARTKEY`,
            `t2`.`L_QUANTITY`,
            `t2`.`L_EXTENDEDPRICE`,
            `t2`.`L_DISCOUNT`,
            `t2`.`L_SHIPINSTRUCT`,
            `t2`.`L_SHIPMODE`
          FROM (
            SELECT
              `L_PARTKEY`,
              `L_QUANTITY`,
              `L_EXTENDEDPRICE`,
              `L_DISCOUNT`,
              `L_SHIPINSTRUCT`,
              `L_SHIPMODE`
            FROM `bigframes-dev.tpch.LINEITEM` FOR SYSTEM_TIME AS OF CAST('2026-03-10T18:00:00' AS DATETIME)
          ) AS `t2`
        ) AS `t8`
          ON COALESCE(`t7`.`P_PARTKEY`, 0) = COALESCE(`t8`.`L_PARTKEY`, 0)
          AND COALESCE(`t7`.`P_PARTKEY`, 1) = COALESCE(`t8`.`L_PARTKEY`, 1)
      ) AS `t9`
      WHERE
        `t9`.`bfuid_col_3746`
    ) AS `t10`
  ) AS `t11`
) AS `t13`
CROSS JOIN (
  SELECT
    *
  FROM (
    SELECT
      *
    FROM UNNEST(ARRAY<STRUCT<`col_1` INT64>>[STRUCT(0)]) AS `col_1`
  ) AS `t0`
) AS `t6`