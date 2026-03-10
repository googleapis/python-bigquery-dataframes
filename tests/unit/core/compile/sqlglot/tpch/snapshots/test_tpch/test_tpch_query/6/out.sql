WITH `bfcte_0` AS (
  SELECT
    `L_QUANTITY`,
    `L_EXTENDEDPRICE`,
    `L_DISCOUNT`,
    `L_SHIPDATE`,
    `L_QUANTITY` AS `bfcol_4`,
    `L_EXTENDEDPRICE` AS `bfcol_5`,
    `L_DISCOUNT` AS `bfcol_6`,
    (
      `L_SHIPDATE` >= CAST('1994-01-01' AS DATE)
    )
    AND (
      `L_SHIPDATE` < CAST('1995-01-01' AS DATE)
    ) AS `bfcol_7`,
    `L_QUANTITY` AS `bfcol_15`,
    `L_EXTENDEDPRICE` AS `bfcol_16`,
    `L_DISCOUNT` AS `bfcol_17`,
    (
      `L_DISCOUNT` >= 0.05
    ) AND (
      `L_DISCOUNT` <= 0.07
    ) AS `bfcol_18`,
    `L_EXTENDEDPRICE` AS `bfcol_26`,
    `L_DISCOUNT` AS `bfcol_27`,
    `L_QUANTITY` < 24 AS `bfcol_28`,
    `L_EXTENDEDPRICE` AS `bfcol_34`,
    `L_DISCOUNT` AS `bfcol_35`,
    `L_EXTENDEDPRICE` * `L_DISCOUNT` AS `bfcol_38`
  FROM `bigframes-dev`.`tpch`.`LINEITEM` AS `bft_0` FOR SYSTEM_TIME AS OF '2026-03-10T18:00:00'
  WHERE
    (
      `L_SHIPDATE` >= CAST('1994-01-01' AS DATE)
    )
    AND (
      `L_SHIPDATE` < CAST('1995-01-01' AS DATE)
    )
    AND (
      `L_DISCOUNT` >= 0.05
    )
    AND (
      `L_DISCOUNT` <= 0.07
    )
    AND `L_QUANTITY` < 24
), `bfcte_1` AS (
  SELECT
    COALESCE(SUM(`bfcol_38`), 0) AS `bfcol_40`
  FROM `bfcte_0`
), `bfcte_2` AS (
  SELECT
    *
  FROM `bfcte_1`
), `bfcte_3` AS (
  SELECT
    *
  FROM UNNEST(ARRAY<STRUCT<`bfcol_41` INT64>>[STRUCT(0)])
), `bfcte_4` AS (
  SELECT
    *
  FROM `bfcte_2`
  CROSS JOIN `bfcte_3`
)
SELECT
  CASE WHEN `bfcol_41` = 0 THEN `bfcol_40` END AS `REVENUE`
FROM `bfcte_4`