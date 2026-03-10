WITH `bfcte_9` AS (
  SELECT
    `L_ORDERKEY` AS `bfcol_7`,
    `L_SUPPKEY` AS `bfcol_8`,
    `L_EXTENDEDPRICE` * (
      1.0 - `L_DISCOUNT`
    ) AS `bfcol_9`
  FROM `bigframes-dev`.`tpch`.`LINEITEM` AS `bft_5` FOR SYSTEM_TIME AS OF '2026-03-10T18:00:00'
), `bfcte_6` AS (
  SELECT
    `O_ORDERKEY` AS `bfcol_19`,
    `O_CUSTKEY` AS `bfcol_20`
  FROM `bigframes-dev`.`tpch`.`ORDERS` AS `bft_4` FOR SYSTEM_TIME AS OF '2026-03-10T18:00:00'
  WHERE
    (
      `O_ORDERDATE` >= CAST('1994-01-01' AS DATE)
    )
    AND (
      `O_ORDERDATE` < CAST('1995-01-01' AS DATE)
    )
), `bfcte_0` AS (
  SELECT
    `R_REGIONKEY` AS `bfcol_27`
  FROM `bigframes-dev`.`tpch`.`REGION` AS `bft_3` FOR SYSTEM_TIME AS OF '2026-03-10T18:00:00'
  WHERE
    `R_NAME` = 'ASIA'
), `bfcte_1` AS (
  SELECT
    `N_NATIONKEY` AS `bfcol_28`,
    `N_NAME` AS `bfcol_29`,
    `N_REGIONKEY` AS `bfcol_30`
  FROM `bigframes-dev`.`tpch`.`NATION` AS `bft_2` FOR SYSTEM_TIME AS OF '2026-03-10T18:00:00'
), `bfcte_2` AS (
  SELECT
    *
  FROM `bfcte_0`
  INNER JOIN `bfcte_1`
    ON COALESCE(`bfcol_27`, 0) = COALESCE(`bfcol_30`, 0)
    AND COALESCE(`bfcol_27`, 1) = COALESCE(`bfcol_30`, 1)
), `bfcte_3` AS (
  SELECT
    `bfcol_28` AS `bfcol_31`,
    `bfcol_29` AS `bfcol_32`
  FROM `bfcte_2`
), `bfcte_4` AS (
  SELECT
    `C_CUSTKEY` AS `bfcol_33`,
    `C_NATIONKEY` AS `bfcol_34`
  FROM `bigframes-dev`.`tpch`.`CUSTOMER` AS `bft_1` FOR SYSTEM_TIME AS OF '2026-03-10T18:00:00'
), `bfcte_5` AS (
  SELECT
    *
  FROM `bfcte_3`
  INNER JOIN `bfcte_4`
    ON COALESCE(`bfcol_31`, 0) = COALESCE(`bfcol_34`, 0)
    AND COALESCE(`bfcol_31`, 1) = COALESCE(`bfcol_34`, 1)
), `bfcte_7` AS (
  SELECT
    `bfcol_31` AS `bfcol_35`,
    `bfcol_32` AS `bfcol_36`,
    `bfcol_33` AS `bfcol_37`
  FROM `bfcte_5`
), `bfcte_8` AS (
  SELECT
    *
  FROM `bfcte_6`
  INNER JOIN `bfcte_7`
    ON COALESCE(`bfcol_20`, 0) = COALESCE(`bfcol_37`, 0)
    AND COALESCE(`bfcol_20`, 1) = COALESCE(`bfcol_37`, 1)
), `bfcte_10` AS (
  SELECT
    `bfcol_19` AS `bfcol_38`,
    `bfcol_35` AS `bfcol_39`,
    `bfcol_36` AS `bfcol_40`
  FROM `bfcte_8`
), `bfcte_11` AS (
  SELECT
    *
  FROM `bfcte_9`
  INNER JOIN `bfcte_10`
    ON COALESCE(`bfcol_7`, 0) = COALESCE(`bfcol_38`, 0)
    AND COALESCE(`bfcol_7`, 1) = COALESCE(`bfcol_38`, 1)
), `bfcte_12` AS (
  SELECT
    `bfcol_8` AS `bfcol_41`,
    `bfcol_9` AS `bfcol_42`,
    `bfcol_39` AS `bfcol_43`,
    `bfcol_40` AS `bfcol_44`
  FROM `bfcte_11`
), `bfcte_13` AS (
  SELECT
    `S_SUPPKEY` AS `bfcol_45`,
    `S_NATIONKEY` AS `bfcol_46`
  FROM `bigframes-dev`.`tpch`.`SUPPLIER` AS `bft_0` FOR SYSTEM_TIME AS OF '2026-03-10T18:00:00'
), `bfcte_14` AS (
  SELECT
    *
  FROM `bfcte_12`
  INNER JOIN `bfcte_13`
    ON COALESCE(`bfcol_41`, 0) = COALESCE(`bfcol_45`, 0)
    AND COALESCE(`bfcol_41`, 1) = COALESCE(`bfcol_45`, 1)
    AND COALESCE(`bfcol_43`, 0) = COALESCE(`bfcol_46`, 0)
    AND COALESCE(`bfcol_43`, 1) = COALESCE(`bfcol_46`, 1)
), `bfcte_15` AS (
  SELECT
    `bfcol_44`,
    COALESCE(SUM(`bfcol_42`), 0) AS `bfcol_49`
  FROM `bfcte_14`
  WHERE
    NOT `bfcol_44` IS NULL
  GROUP BY
    `bfcol_44`
)
SELECT
  `bfcol_44` AS `N_NAME`,
  `bfcol_49` AS `REVENUE`
FROM `bfcte_15`
ORDER BY
  `bfcol_49` DESC,
  `bfcol_44` ASC NULLS LAST