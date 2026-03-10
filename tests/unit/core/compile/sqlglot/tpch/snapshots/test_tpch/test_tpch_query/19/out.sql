WITH `bfcte_0` AS (
  SELECT
    `P_PARTKEY` AS `bfcol_0`,
    `P_BRAND` AS `bfcol_1`,
    `P_SIZE` AS `bfcol_2`,
    `P_CONTAINER` AS `bfcol_3`
  FROM `bigframes-dev`.`tpch`.`PART` AS `bft_1` FOR SYSTEM_TIME AS OF '2026-03-10T18:00:00'
), `bfcte_1` AS (
  SELECT
    `L_PARTKEY` AS `bfcol_4`,
    `L_QUANTITY` AS `bfcol_5`,
    `L_EXTENDEDPRICE` AS `bfcol_6`,
    `L_DISCOUNT` AS `bfcol_7`,
    `L_SHIPINSTRUCT` AS `bfcol_8`,
    `L_SHIPMODE` AS `bfcol_9`
  FROM `bigframes-dev`.`tpch`.`LINEITEM` AS `bft_0` FOR SYSTEM_TIME AS OF '2026-03-10T18:00:00'
), `bfcte_2` AS (
  SELECT
    *
  FROM `bfcte_0`
  INNER JOIN `bfcte_1`
    ON COALESCE(`bfcol_0`, 0) = COALESCE(`bfcol_4`, 0)
    AND COALESCE(`bfcol_0`, 1) = COALESCE(`bfcol_4`, 1)
), `bfcte_3` AS (
  SELECT
    `bfcol_0`,
    `bfcol_1`,
    `bfcol_2`,
    `bfcol_3`,
    `bfcol_4`,
    `bfcol_5`,
    `bfcol_6`,
    `bfcol_7`,
    `bfcol_8`,
    `bfcol_9`,
    `bfcol_6` AS `bfcol_18`,
    `bfcol_7` AS `bfcol_19`,
    (
      COALESCE(COALESCE(`bfcol_9` IN ('AIR', 'AIR REG'), FALSE), FALSE)
      AND (
        `bfcol_8` = 'DELIVER IN PERSON'
      )
    )
    AND (
      (
        (
          (
            (
              (
                `bfcol_1` = 'Brand#12'
              )
              AND COALESCE(COALESCE(`bfcol_3` IN ('SM CASE', 'SM BOX', 'SM PACK', 'SM PKG'), FALSE), FALSE)
            )
            AND (
              (
                `bfcol_5` >= 1
              ) AND (
                `bfcol_5` <= 11
              )
            )
          )
          AND (
            (
              `bfcol_2` >= 1
            ) AND (
              `bfcol_2` <= 5
            )
          )
        )
        OR (
          (
            (
              (
                `bfcol_1` = 'Brand#23'
              )
              AND COALESCE(COALESCE(`bfcol_3` IN ('MED BAG', 'MED BOX', 'MED PKG', 'MED PACK'), FALSE), FALSE)
            )
            AND (
              (
                `bfcol_5` >= 10
              ) AND (
                `bfcol_5` <= 20
              )
            )
          )
          AND (
            (
              `bfcol_2` >= 1
            ) AND (
              `bfcol_2` <= 10
            )
          )
        )
      )
      OR (
        (
          (
            (
              `bfcol_1` = 'Brand#34'
            )
            AND COALESCE(COALESCE(`bfcol_3` IN ('LG CASE', 'LG BOX', 'LG PACK', 'LG PKG'), FALSE), FALSE)
          )
          AND (
            (
              `bfcol_5` >= 20
            ) AND (
              `bfcol_5` <= 30
            )
          )
        )
        AND (
          (
            `bfcol_2` >= 1
          ) AND (
            `bfcol_2` <= 15
          )
        )
      )
    ) AS `bfcol_20`,
    `bfcol_6` AS `bfcol_26`,
    1 - `bfcol_7` AS `bfcol_27`,
    `bfcol_6` * (
      1 - `bfcol_7`
    ) AS `bfcol_30`
  FROM `bfcte_2`
  WHERE
    (
      COALESCE(COALESCE(`bfcol_9` IN ('AIR', 'AIR REG'), FALSE), FALSE)
      AND (
        `bfcol_8` = 'DELIVER IN PERSON'
      )
    )
    AND (
      (
        (
          (
            (
              (
                `bfcol_1` = 'Brand#12'
              )
              AND COALESCE(COALESCE(`bfcol_3` IN ('SM CASE', 'SM BOX', 'SM PACK', 'SM PKG'), FALSE), FALSE)
            )
            AND (
              (
                `bfcol_5` >= 1
              ) AND (
                `bfcol_5` <= 11
              )
            )
          )
          AND (
            (
              `bfcol_2` >= 1
            ) AND (
              `bfcol_2` <= 5
            )
          )
        )
        OR (
          (
            (
              (
                `bfcol_1` = 'Brand#23'
              )
              AND COALESCE(COALESCE(`bfcol_3` IN ('MED BAG', 'MED BOX', 'MED PKG', 'MED PACK'), FALSE), FALSE)
            )
            AND (
              (
                `bfcol_5` >= 10
              ) AND (
                `bfcol_5` <= 20
              )
            )
          )
          AND (
            (
              `bfcol_2` >= 1
            ) AND (
              `bfcol_2` <= 10
            )
          )
        )
      )
      OR (
        (
          (
            (
              `bfcol_1` = 'Brand#34'
            )
            AND COALESCE(COALESCE(`bfcol_3` IN ('LG CASE', 'LG BOX', 'LG PACK', 'LG PKG'), FALSE), FALSE)
          )
          AND (
            (
              `bfcol_5` >= 20
            ) AND (
              `bfcol_5` <= 30
            )
          )
        )
        AND (
          (
            `bfcol_2` >= 1
          ) AND (
            `bfcol_2` <= 15
          )
        )
      )
    )
), `bfcte_4` AS (
  SELECT
    COALESCE(SUM(`bfcol_30`), 0) AS `bfcol_32`
  FROM `bfcte_3`
), `bfcte_5` AS (
  SELECT
    *
  FROM `bfcte_4`
), `bfcte_6` AS (
  SELECT
    *
  FROM UNNEST(ARRAY<STRUCT<`bfcol_33` INT64>>[STRUCT(0)])
), `bfcte_7` AS (
  SELECT
    *
  FROM `bfcte_5`
  CROSS JOIN `bfcte_6`
)
SELECT
  CASE WHEN `bfcol_33` = 0 THEN `bfcol_32` END AS `REVENUE`
FROM `bfcte_7`