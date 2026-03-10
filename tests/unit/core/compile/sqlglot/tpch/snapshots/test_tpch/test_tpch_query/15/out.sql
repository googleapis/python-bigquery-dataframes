WITH `bfcte_6` AS (
  SELECT
    `S_SUPPKEY` AS `bfcol_0`,
    `S_NAME` AS `bfcol_1`,
    `S_ADDRESS` AS `bfcol_2`,
    `S_PHONE` AS `bfcol_3`
  FROM `bigframes-dev`.`tpch`.`SUPPLIER` AS `bft_1` FOR SYSTEM_TIME AS OF '2026-03-10T18:00:00'
), `bfcte_1` AS (
  SELECT
    `L_SUPPKEY`,
    `L_EXTENDEDPRICE`,
    `L_DISCOUNT`,
    `L_SHIPDATE`,
    `L_SUPPKEY` AS `bfcol_8`,
    `L_EXTENDEDPRICE` AS `bfcol_9`,
    `L_DISCOUNT` AS `bfcol_10`,
    (
      `L_SHIPDATE` >= CAST('1996-01-01' AS DATE)
    )
    AND (
      `L_SHIPDATE` < CAST('1996-04-01' AS DATE)
    ) AS `bfcol_11`,
    `L_SUPPKEY` AS `bfcol_19`,
    `L_EXTENDEDPRICE` * (
      1 - `L_DISCOUNT`
    ) AS `bfcol_20`
  FROM `bigframes-dev`.`tpch`.`LINEITEM` AS `bft_0` FOR SYSTEM_TIME AS OF '2026-03-10T18:00:00'
  WHERE
    (
      `L_SHIPDATE` >= CAST('1996-01-01' AS DATE)
    )
    AND (
      `L_SHIPDATE` < CAST('1996-04-01' AS DATE)
    )
), `bfcte_3` AS (
  SELECT
    `bfcol_19`,
    COALESCE(SUM(`bfcol_20`), 0) AS `bfcol_23`
  FROM `bfcte_1`
  WHERE
    NOT `bfcol_19` IS NULL
  GROUP BY
    `bfcol_19`
), `bfcte_7` AS (
  SELECT
    `bfcol_19` AS `bfcol_26`,
    ROUND(`bfcol_23`, 2) AS `bfcol_27`
  FROM `bfcte_3`
), `bfcte_9` AS (
  SELECT
    *
  FROM `bfcte_6`
  INNER JOIN `bfcte_7`
    ON `bfcol_0` = `bfcol_26`
), `bfcte_16` AS (
  SELECT
    `bfcol_0` AS `bfcol_28`,
    `bfcol_1` AS `bfcol_29`,
    `bfcol_2` AS `bfcol_30`,
    `bfcol_3` AS `bfcol_31`,
    `bfcol_27` AS `bfcol_32`
  FROM `bfcte_9`
), `bfcte_11` AS (
  SELECT
    *
  FROM UNNEST(ARRAY<STRUCT<`bfcol_33` STRING, `bfcol_34` INT64, `bfcol_35` INT64>>[STRUCT('TOTAL_REVENUE', 0, 0)])
), `bfcte_4` AS (
  SELECT
    `S_SUPPKEY` AS `bfcol_36`
  FROM `bigframes-dev`.`tpch`.`SUPPLIER` AS `bft_1` FOR SYSTEM_TIME AS OF '2026-03-10T18:00:00'
), `bfcte_0` AS (
  SELECT
    `L_SUPPKEY`,
    `L_EXTENDEDPRICE`,
    `L_DISCOUNT`,
    `L_SHIPDATE`,
    `L_SUPPKEY` AS `bfcol_41`,
    `L_EXTENDEDPRICE` AS `bfcol_42`,
    `L_DISCOUNT` AS `bfcol_43`,
    (
      `L_SHIPDATE` >= CAST('1996-01-01' AS DATE)
    )
    AND (
      `L_SHIPDATE` < CAST('1996-04-01' AS DATE)
    ) AS `bfcol_44`,
    `L_SUPPKEY` AS `bfcol_52`,
    `L_EXTENDEDPRICE` * (
      1 - `L_DISCOUNT`
    ) AS `bfcol_53`
  FROM `bigframes-dev`.`tpch`.`LINEITEM` AS `bft_0` FOR SYSTEM_TIME AS OF '2026-03-10T18:00:00'
  WHERE
    (
      `L_SHIPDATE` >= CAST('1996-01-01' AS DATE)
    )
    AND (
      `L_SHIPDATE` < CAST('1996-04-01' AS DATE)
    )
), `bfcte_2` AS (
  SELECT
    `bfcol_52`,
    COALESCE(SUM(`bfcol_53`), 0) AS `bfcol_56`
  FROM `bfcte_0`
  WHERE
    NOT `bfcol_52` IS NULL
  GROUP BY
    `bfcol_52`
), `bfcte_5` AS (
  SELECT
    `bfcol_52` AS `bfcol_59`,
    ROUND(`bfcol_56`, 2) AS `bfcol_60`
  FROM `bfcte_2`
), `bfcte_8` AS (
  SELECT
    *
  FROM `bfcte_4`
  INNER JOIN `bfcte_5`
    ON `bfcol_36` = `bfcol_59`
), `bfcte_10` AS (
  SELECT
    MAX(`bfcol_60`) AS `bfcol_62`
  FROM `bfcte_8`
), `bfcte_12` AS (
  SELECT
    `bfcol_62`,
    0 AS `bfcol_63`
  FROM `bfcte_10`
), `bfcte_13` AS (
  SELECT
    *
  FROM `bfcte_11`
  CROSS JOIN `bfcte_12`
), `bfcte_14` AS (
  SELECT
    `bfcol_33`,
    `bfcol_34`,
    `bfcol_35`,
    `bfcol_62`,
    `bfcol_63`,
    CASE WHEN `bfcol_35` = 0 THEN `bfcol_62` END AS `bfcol_64`,
    IF(`bfcol_63` = 0, CASE WHEN `bfcol_35` = 0 THEN `bfcol_62` END, NULL) AS `bfcol_69`
  FROM `bfcte_13`
), `bfcte_15` AS (
  SELECT
    `bfcol_33`,
    `bfcol_34`,
    ANY_VALUE(`bfcol_69`) AS `bfcol_73`
  FROM `bfcte_14`
  WHERE
    NOT `bfcol_33` IS NULL AND NOT `bfcol_34` IS NULL
  GROUP BY
    `bfcol_33`,
    `bfcol_34`
), `bfcte_17` AS (
  SELECT
    `bfcol_73` AS `bfcol_74`
  FROM `bfcte_15`
), `bfcte_18` AS (
  SELECT
    *
  FROM `bfcte_16`
  CROSS JOIN `bfcte_17`
)
SELECT
  `bfcol_28` AS `S_SUPPKEY`,
  `bfcol_29` AS `S_NAME`,
  `bfcol_30` AS `S_ADDRESS`,
  `bfcol_31` AS `S_PHONE`,
  `bfcol_32` AS `TOTAL_REVENUE`
FROM `bfcte_18`
WHERE
  `bfcol_32` = `bfcol_74`
ORDER BY
  `bfcol_28` ASC NULLS LAST