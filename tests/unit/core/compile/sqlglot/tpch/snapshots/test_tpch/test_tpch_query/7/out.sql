WITH `bfcte_0` AS (
  SELECT
    `C_CUSTKEY` AS `bfcol_0`,
    `C_NATIONKEY` AS `bfcol_1`
  FROM `bigframes-dev`.`tpch`.`CUSTOMER` AS `bft_4` FOR SYSTEM_TIME AS OF '2026-03-10T18:00:00'
), `bfcte_1` AS (
  SELECT
    `N_NATIONKEY` AS `bfcol_10`,
    `N_NAME` AS `bfcol_11`
  FROM `bigframes-dev`.`tpch`.`NATION` AS `bft_0` FOR SYSTEM_TIME AS OF '2026-03-10T18:00:00'
  WHERE
    COALESCE(COALESCE(`N_NAME` IN ('FRANCE', 'GERMANY'), FALSE), FALSE)
), `bfcte_2` AS (
  SELECT
    *
  FROM `bfcte_0`
  INNER JOIN `bfcte_1`
    ON COALESCE(`bfcol_1`, 0) = COALESCE(`bfcol_10`, 0)
    AND COALESCE(`bfcol_1`, 1) = COALESCE(`bfcol_10`, 1)
), `bfcte_3` AS (
  SELECT
    `bfcol_0` AS `bfcol_12`,
    `bfcol_11` AS `bfcol_13`
  FROM `bfcte_2`
), `bfcte_4` AS (
  SELECT
    `O_ORDERKEY` AS `bfcol_14`,
    `O_CUSTKEY` AS `bfcol_15`
  FROM `bigframes-dev`.`tpch`.`ORDERS` AS `bft_3` FOR SYSTEM_TIME AS OF '2026-03-10T18:00:00'
), `bfcte_5` AS (
  SELECT
    *
  FROM `bfcte_3`
  INNER JOIN `bfcte_4`
    ON COALESCE(`bfcol_12`, 0) = COALESCE(`bfcol_15`, 0)
    AND COALESCE(`bfcol_12`, 1) = COALESCE(`bfcol_15`, 1)
), `bfcte_6` AS (
  SELECT
    `bfcol_13` AS `bfcol_16`,
    `bfcol_14` AS `bfcol_17`
  FROM `bfcte_5`
), `bfcte_7` AS (
  SELECT
    `L_ORDERKEY` AS `bfcol_35`,
    `L_SUPPKEY` AS `bfcol_36`,
    `L_EXTENDEDPRICE` AS `bfcol_37`,
    `L_DISCOUNT` AS `bfcol_38`,
    `L_SHIPDATE` AS `bfcol_39`
  FROM `bigframes-dev`.`tpch`.`LINEITEM` AS `bft_2` FOR SYSTEM_TIME AS OF '2026-03-10T18:00:00'
  WHERE
    (
      `L_SHIPDATE` >= CAST('1995-01-01' AS DATE)
    )
    AND (
      `L_SHIPDATE` <= CAST('1996-12-31' AS DATE)
    )
), `bfcte_8` AS (
  SELECT
    *
  FROM `bfcte_6`
  INNER JOIN `bfcte_7`
    ON COALESCE(`bfcol_17`, 0) = COALESCE(`bfcol_35`, 0)
    AND COALESCE(`bfcol_17`, 1) = COALESCE(`bfcol_35`, 1)
), `bfcte_9` AS (
  SELECT
    `bfcol_16` AS `bfcol_40`,
    `bfcol_36` AS `bfcol_41`,
    `bfcol_37` AS `bfcol_42`,
    `bfcol_38` AS `bfcol_43`,
    `bfcol_39` AS `bfcol_44`
  FROM `bfcte_8`
), `bfcte_10` AS (
  SELECT
    `S_SUPPKEY` AS `bfcol_45`,
    `S_NATIONKEY` AS `bfcol_46`
  FROM `bigframes-dev`.`tpch`.`SUPPLIER` AS `bft_1` FOR SYSTEM_TIME AS OF '2026-03-10T18:00:00'
), `bfcte_11` AS (
  SELECT
    *
  FROM `bfcte_9`
  INNER JOIN `bfcte_10`
    ON COALESCE(`bfcol_41`, 0) = COALESCE(`bfcol_45`, 0)
    AND COALESCE(`bfcol_41`, 1) = COALESCE(`bfcol_45`, 1)
), `bfcte_12` AS (
  SELECT
    `bfcol_40` AS `bfcol_47`,
    `bfcol_42` AS `bfcol_48`,
    `bfcol_43` AS `bfcol_49`,
    `bfcol_44` AS `bfcol_50`,
    `bfcol_46` AS `bfcol_51`
  FROM `bfcte_11`
), `bfcte_13` AS (
  SELECT
    `N_NATIONKEY` AS `bfcol_60`,
    `N_NAME` AS `bfcol_61`
  FROM `bigframes-dev`.`tpch`.`NATION` AS `bft_0` FOR SYSTEM_TIME AS OF '2026-03-10T18:00:00'
  WHERE
    COALESCE(COALESCE(`N_NAME` IN ('FRANCE', 'GERMANY'), FALSE), FALSE)
), `bfcte_14` AS (
  SELECT
    *
  FROM `bfcte_12`
  INNER JOIN `bfcte_13`
    ON COALESCE(`bfcol_51`, 0) = COALESCE(`bfcol_60`, 0)
    AND COALESCE(`bfcol_51`, 1) = COALESCE(`bfcol_60`, 1)
), `bfcte_15` AS (
  SELECT
    `bfcol_47`,
    `bfcol_48`,
    `bfcol_49`,
    `bfcol_50`,
    `bfcol_51`,
    `bfcol_60`,
    `bfcol_61`,
    `bfcol_47` AS `bfcol_67`,
    `bfcol_48` AS `bfcol_68`,
    `bfcol_49` AS `bfcol_69`,
    `bfcol_50` AS `bfcol_70`,
    `bfcol_61` AS `bfcol_71`,
    `bfcol_47` <> `bfcol_61` AS `bfcol_72`,
    `bfcol_47` AS `bfcol_84`,
    `bfcol_50` AS `bfcol_85`,
    `bfcol_61` AS `bfcol_86`,
    `bfcol_48` * (
      1.0 - `bfcol_49`
    ) AS `bfcol_87`,
    `bfcol_47` AS `bfcol_92`,
    `bfcol_61` AS `bfcol_93`,
    `bfcol_48` * (
      1.0 - `bfcol_49`
    ) AS `bfcol_94`,
    EXTRACT(YEAR FROM `bfcol_50`) AS `bfcol_95`
  FROM `bfcte_14`
  WHERE
    `bfcol_47` <> `bfcol_61`
), `bfcte_16` AS (
  SELECT
    `bfcol_93`,
    `bfcol_92`,
    `bfcol_95`,
    COALESCE(SUM(`bfcol_94`), 0) AS `bfcol_100`
  FROM `bfcte_15`
  WHERE
    NOT `bfcol_93` IS NULL AND NOT `bfcol_92` IS NULL AND NOT `bfcol_95` IS NULL
  GROUP BY
    `bfcol_93`,
    `bfcol_92`,
    `bfcol_95`
)
SELECT
  `bfcol_93` AS `SUPP_NATION`,
  `bfcol_92` AS `CUST_NATION`,
  `bfcol_95` AS `L_YEAR`,
  `bfcol_100` AS `REVENUE`
FROM `bfcte_16`
ORDER BY
  `bfcol_93` ASC NULLS LAST,
  `bfcol_92` ASC NULLS LAST,
  `bfcol_95` ASC NULLS LAST