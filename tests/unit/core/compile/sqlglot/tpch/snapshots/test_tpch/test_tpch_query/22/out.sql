WITH `bfcte_9` AS (
  SELECT
    `C_CUSTKEY` AS `bfcol_17`,
    `C_ACCTBAL` AS `bfcol_18`,
    SUBSTRING(`C_PHONE`, 1, 2) AS `bfcol_19`
  FROM `bigframes-dev`.`tpch`.`CUSTOMER` AS `bft_1` FOR SYSTEM_TIME AS OF '2026-03-10T18:00:00'
  WHERE
    COALESCE(
      COALESCE(SUBSTRING(`C_PHONE`, 1, 2) IN ('13', '31', '23', '29', '30', '18', '17'), FALSE),
      FALSE
    )
), `bfcte_4` AS (
  SELECT
    *
  FROM UNNEST(ARRAY<STRUCT<`bfcol_20` STRING, `bfcol_21` INT64, `bfcol_22` INT64>>[STRUCT('C_ACCTBAL', 0, 0)])
), `bfcte_1` AS (
  SELECT
    `C_PHONE`,
    `C_ACCTBAL`,
    `C_ACCTBAL` AS `bfcol_25`,
    SUBSTRING(`C_PHONE`, 1, 2) AS `bfcol_26`,
    `C_ACCTBAL` AS `bfcol_29`,
    COALESCE(
      COALESCE(SUBSTRING(`C_PHONE`, 1, 2) IN ('13', '31', '23', '29', '30', '18', '17'), FALSE),
      FALSE
    ) AS `bfcol_30`,
    `C_ACCTBAL` AS `bfcol_34`,
    `C_ACCTBAL` > 0.0 AS `bfcol_35`
  FROM `bigframes-dev`.`tpch`.`CUSTOMER` AS `bft_1` FOR SYSTEM_TIME AS OF '2026-03-10T18:00:00'
  WHERE
    COALESCE(
      COALESCE(SUBSTRING(`C_PHONE`, 1, 2) IN ('13', '31', '23', '29', '30', '18', '17'), FALSE),
      FALSE
    )
    AND `C_ACCTBAL` > 0.0
), `bfcte_3` AS (
  SELECT
    AVG(`bfcol_34`) AS `bfcol_39`
  FROM `bfcte_1`
), `bfcte_5` AS (
  SELECT
    `bfcol_39`,
    0 AS `bfcol_40`
  FROM `bfcte_3`
), `bfcte_6` AS (
  SELECT
    *
  FROM `bfcte_4`
  CROSS JOIN `bfcte_5`
), `bfcte_7` AS (
  SELECT
    `bfcol_20`,
    `bfcol_21`,
    `bfcol_22`,
    `bfcol_39`,
    `bfcol_40`,
    CASE WHEN `bfcol_22` = 0 THEN `bfcol_39` END AS `bfcol_41`,
    IF(`bfcol_40` = 0, CASE WHEN `bfcol_22` = 0 THEN `bfcol_39` END, NULL) AS `bfcol_46`
  FROM `bfcte_6`
), `bfcte_8` AS (
  SELECT
    `bfcol_20`,
    `bfcol_21`,
    ANY_VALUE(`bfcol_46`) AS `bfcol_50`
  FROM `bfcte_7`
  WHERE
    NOT `bfcol_20` IS NULL AND NOT `bfcol_21` IS NULL
  GROUP BY
    `bfcol_20`,
    `bfcol_21`
), `bfcte_10` AS (
  SELECT
    `bfcol_50` AS `bfcol_51`
  FROM `bfcte_8`
), `bfcte_11` AS (
  SELECT
    *
  FROM `bfcte_9`
  CROSS JOIN `bfcte_10`
), `bfcte_12` AS (
  SELECT
    `bfcol_17` AS `bfcol_60`,
    `bfcol_18` AS `bfcol_61`,
    `bfcol_19` AS `bfcol_62`
  FROM `bfcte_11`
  WHERE
    `bfcol_18` > `bfcol_51`
), `bfcte_0` AS (
  SELECT
    `O_CUSTKEY`
  FROM `bigframes-dev`.`tpch`.`ORDERS` AS `bft_0` FOR SYSTEM_TIME AS OF '2026-03-10T18:00:00'
), `bfcte_2` AS (
  SELECT
    `O_CUSTKEY`
  FROM `bfcte_0`
  GROUP BY
    `O_CUSTKEY`
), `bfcte_13` AS (
  SELECT
    `bfcte_12`.*,
    EXISTS(
      SELECT
        1
      FROM (
        SELECT
          `O_CUSTKEY` AS `bfcol_63`
        FROM `bfcte_2`
      ) AS `bft_2`
      WHERE
        COALESCE(`bfcte_12`.`bfcol_60`, 0) = COALESCE(`bft_2`.`bfcol_63`, 0)
        AND COALESCE(`bfcte_12`.`bfcol_60`, 1) = COALESCE(`bft_2`.`bfcol_63`, 1)
    ) AS `bfcol_64`
  FROM `bfcte_12`
), `bfcte_14` AS (
  SELECT
    `bfcol_60`,
    `bfcol_61`,
    `bfcol_62`,
    `bfcol_64`,
    NOT (
      `bfcol_64`
    ) AS `bfcol_65`
  FROM `bfcte_13`
  WHERE
    NOT (
      `bfcol_64`
    )
), `bfcte_15` AS (
  SELECT
    `bfcol_62`,
    COUNT(`bfcol_60`) AS `bfcol_73`,
    COALESCE(SUM(`bfcol_61`), 0) AS `bfcol_74`
  FROM `bfcte_14`
  WHERE
    NOT `bfcol_62` IS NULL
  GROUP BY
    `bfcol_62`
)
SELECT
  `bfcol_62` AS `CNTRYCODE`,
  `bfcol_73` AS `NUMCUST`,
  `bfcol_74` AS `TOTACCTBAL`
FROM `bfcte_15`
ORDER BY
  `bfcol_62` ASC NULLS LAST