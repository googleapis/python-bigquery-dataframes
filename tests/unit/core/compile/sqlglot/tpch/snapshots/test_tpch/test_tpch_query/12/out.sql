WITH `bfcte_0` AS (
  SELECT
    `O_ORDERKEY` AS `bfcol_0`,
    `O_ORDERPRIORITY` AS `bfcol_1`
  FROM `bigframes-dev`.`tpch`.`ORDERS` AS `bft_1` FOR SYSTEM_TIME AS OF '2026-03-10T18:00:00'
), `bfcte_1` AS (
  SELECT
    `L_ORDERKEY` AS `bfcol_2`,
    `L_SHIPDATE` AS `bfcol_3`,
    `L_COMMITDATE` AS `bfcol_4`,
    `L_RECEIPTDATE` AS `bfcol_5`,
    `L_SHIPMODE` AS `bfcol_6`
  FROM `bigframes-dev`.`tpch`.`LINEITEM` AS `bft_0` FOR SYSTEM_TIME AS OF '2026-03-10T18:00:00'
), `bfcte_2` AS (
  SELECT
    *
  FROM `bfcte_0`
  INNER JOIN `bfcte_1`
    ON COALESCE(`bfcol_0`, 0) = COALESCE(`bfcol_2`, 0)
    AND COALESCE(`bfcol_0`, 1) = COALESCE(`bfcol_2`, 1)
), `bfcte_3` AS (
  SELECT
    `bfcol_0`,
    `bfcol_1`,
    `bfcol_2`,
    `bfcol_3`,
    `bfcol_4`,
    `bfcol_5`,
    `bfcol_6`,
    `bfcol_1` AS `bfcol_12`,
    `bfcol_6` AS `bfcol_13`,
    (
      (
        (
          COALESCE(COALESCE(`bfcol_6` IN ('MAIL', 'SHIP'), FALSE), FALSE)
          AND (
            `bfcol_4` < `bfcol_5`
          )
        )
        AND (
          `bfcol_3` < `bfcol_4`
        )
      )
      AND (
        `bfcol_5` >= CAST('1994-01-01' AS DATE)
      )
    )
    AND (
      `bfcol_5` < CAST('1995-01-01' AS DATE)
    ) AS `bfcol_14`,
    `bfcol_1` AS `bfcol_20`,
    `bfcol_6` AS `bfcol_21`,
    CAST(COALESCE(COALESCE(`bfcol_1` IN ('1-URGENT', '2-HIGH'), FALSE), FALSE) AS INT64) AS `bfcol_22`,
    `bfcol_6` AS `bfcol_26`,
    CAST(COALESCE(COALESCE(`bfcol_1` IN ('1-URGENT', '2-HIGH'), FALSE), FALSE) AS INT64) AS `bfcol_27`,
    CAST(NOT (
      COALESCE(COALESCE(`bfcol_1` IN ('1-URGENT', '2-HIGH'), FALSE), FALSE)
    ) AS INT64) AS `bfcol_28`
  FROM `bfcte_2`
  WHERE
    (
      (
        (
          COALESCE(COALESCE(`bfcol_6` IN ('MAIL', 'SHIP'), FALSE), FALSE)
          AND (
            `bfcol_4` < `bfcol_5`
          )
        )
        AND (
          `bfcol_3` < `bfcol_4`
        )
      )
      AND (
        `bfcol_5` >= CAST('1994-01-01' AS DATE)
      )
    )
    AND (
      `bfcol_5` < CAST('1995-01-01' AS DATE)
    )
), `bfcte_4` AS (
  SELECT
    `bfcol_26`,
    COALESCE(SUM(`bfcol_27`), 0) AS `bfcol_32`,
    COALESCE(SUM(`bfcol_28`), 0) AS `bfcol_33`
  FROM `bfcte_3`
  WHERE
    NOT `bfcol_26` IS NULL
  GROUP BY
    `bfcol_26`
)
SELECT
  `bfcol_26` AS `L_SHIPMODE`,
  `bfcol_32` AS `HIGH_LINE_COUNT`,
  `bfcol_33` AS `LOW_LINE_COUNT`
FROM `bfcte_4`
ORDER BY
  `bfcol_26` ASC NULLS LAST