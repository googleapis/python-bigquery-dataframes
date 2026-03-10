WITH `bfcte_0` AS (
  SELECT
    `P_PARTKEY` AS `bfcol_0`,
    `P_NAME` AS `bfcol_1`
  FROM `bigframes-dev`.`tpch`.`PART` AS `bft_5` FOR SYSTEM_TIME AS OF '2026-03-10T18:00:00'
), `bfcte_1` AS (
  SELECT
    `L_ORDERKEY` AS `bfcol_2`,
    `L_PARTKEY` AS `bfcol_3`,
    `L_SUPPKEY` AS `bfcol_4`,
    `L_QUANTITY` AS `bfcol_5`,
    `L_EXTENDEDPRICE` AS `bfcol_6`,
    `L_DISCOUNT` AS `bfcol_7`
  FROM `bigframes-dev`.`tpch`.`LINEITEM` AS `bft_4` FOR SYSTEM_TIME AS OF '2026-03-10T18:00:00'
), `bfcte_2` AS (
  SELECT
    *
  FROM `bfcte_0`
  INNER JOIN `bfcte_1`
    ON COALESCE(`bfcol_0`, 0) = COALESCE(`bfcol_3`, 0)
    AND COALESCE(`bfcol_0`, 1) = COALESCE(`bfcol_3`, 1)
), `bfcte_3` AS (
  SELECT
    `bfcol_1` AS `bfcol_8`,
    `bfcol_2` AS `bfcol_9`,
    `bfcol_3` AS `bfcol_10`,
    `bfcol_4` AS `bfcol_11`,
    `bfcol_5` AS `bfcol_12`,
    `bfcol_6` AS `bfcol_13`,
    `bfcol_7` AS `bfcol_14`
  FROM `bfcte_2`
), `bfcte_4` AS (
  SELECT
    `PS_PARTKEY` AS `bfcol_15`,
    `PS_SUPPKEY` AS `bfcol_16`,
    `PS_SUPPLYCOST` AS `bfcol_17`
  FROM `bigframes-dev`.`tpch`.`PARTSUPP` AS `bft_3` FOR SYSTEM_TIME AS OF '2026-03-10T18:00:00'
), `bfcte_5` AS (
  SELECT
    *
  FROM `bfcte_3`
  INNER JOIN `bfcte_4`
    ON COALESCE(`bfcol_11`, 0) = COALESCE(`bfcol_16`, 0)
    AND COALESCE(`bfcol_11`, 1) = COALESCE(`bfcol_16`, 1)
    AND COALESCE(`bfcol_10`, 0) = COALESCE(`bfcol_15`, 0)
    AND COALESCE(`bfcol_10`, 1) = COALESCE(`bfcol_15`, 1)
), `bfcte_6` AS (
  SELECT
    `bfcol_8` AS `bfcol_18`,
    `bfcol_9` AS `bfcol_19`,
    `bfcol_11` AS `bfcol_20`,
    `bfcol_12` AS `bfcol_21`,
    `bfcol_13` AS `bfcol_22`,
    `bfcol_14` AS `bfcol_23`,
    `bfcol_17` AS `bfcol_24`
  FROM `bfcte_5`
), `bfcte_7` AS (
  SELECT
    `S_SUPPKEY` AS `bfcol_25`,
    `S_NATIONKEY` AS `bfcol_26`
  FROM `bigframes-dev`.`tpch`.`SUPPLIER` AS `bft_2` FOR SYSTEM_TIME AS OF '2026-03-10T18:00:00'
), `bfcte_8` AS (
  SELECT
    *
  FROM `bfcte_6`
  INNER JOIN `bfcte_7`
    ON COALESCE(`bfcol_20`, 0) = COALESCE(`bfcol_25`, 0)
    AND COALESCE(`bfcol_20`, 1) = COALESCE(`bfcol_25`, 1)
), `bfcte_9` AS (
  SELECT
    `bfcol_18` AS `bfcol_27`,
    `bfcol_19` AS `bfcol_28`,
    `bfcol_21` AS `bfcol_29`,
    `bfcol_22` AS `bfcol_30`,
    `bfcol_23` AS `bfcol_31`,
    `bfcol_24` AS `bfcol_32`,
    `bfcol_26` AS `bfcol_33`
  FROM `bfcte_8`
), `bfcte_10` AS (
  SELECT
    `O_ORDERKEY` AS `bfcol_34`,
    `O_ORDERDATE` AS `bfcol_35`
  FROM `bigframes-dev`.`tpch`.`ORDERS` AS `bft_1` FOR SYSTEM_TIME AS OF '2026-03-10T18:00:00'
), `bfcte_11` AS (
  SELECT
    *
  FROM `bfcte_9`
  INNER JOIN `bfcte_10`
    ON COALESCE(`bfcol_28`, 0) = COALESCE(`bfcol_34`, 0)
    AND COALESCE(`bfcol_28`, 1) = COALESCE(`bfcol_34`, 1)
), `bfcte_12` AS (
  SELECT
    `bfcol_27` AS `bfcol_36`,
    `bfcol_29` AS `bfcol_37`,
    `bfcol_30` AS `bfcol_38`,
    `bfcol_31` AS `bfcol_39`,
    `bfcol_32` AS `bfcol_40`,
    `bfcol_33` AS `bfcol_41`,
    `bfcol_35` AS `bfcol_42`
  FROM `bfcte_11`
), `bfcte_13` AS (
  SELECT
    `N_NATIONKEY` AS `bfcol_43`,
    `N_NAME` AS `bfcol_44`
  FROM `bigframes-dev`.`tpch`.`NATION` AS `bft_0` FOR SYSTEM_TIME AS OF '2026-03-10T18:00:00'
), `bfcte_14` AS (
  SELECT
    *
  FROM `bfcte_12`
  INNER JOIN `bfcte_13`
    ON COALESCE(`bfcol_41`, 0) = COALESCE(`bfcol_43`, 0)
    AND COALESCE(`bfcol_41`, 1) = COALESCE(`bfcol_43`, 1)
), `bfcte_15` AS (
  SELECT
    `bfcol_36`,
    `bfcol_37`,
    `bfcol_38`,
    `bfcol_39`,
    `bfcol_40`,
    `bfcol_41`,
    `bfcol_42`,
    `bfcol_43`,
    `bfcol_44`,
    `bfcol_37` AS `bfcol_52`,
    `bfcol_38` AS `bfcol_53`,
    `bfcol_39` AS `bfcol_54`,
    `bfcol_40` AS `bfcol_55`,
    `bfcol_42` AS `bfcol_56`,
    `bfcol_44` AS `bfcol_57`,
    REGEXP_CONTAINS(`bfcol_36`, 'green') AS `bfcol_58`,
    `bfcol_37` AS `bfcol_72`,
    `bfcol_38` AS `bfcol_73`,
    `bfcol_39` AS `bfcol_74`,
    `bfcol_40` AS `bfcol_75`,
    `bfcol_44` AS `bfcol_76`,
    EXTRACT(YEAR FROM `bfcol_42`) AS `bfcol_77`,
    `bfcol_44` AS `bfcol_84`,
    EXTRACT(YEAR FROM `bfcol_42`) AS `bfcol_85`,
    (
      `bfcol_38` * (
        1 - `bfcol_39`
      )
    ) - (
      `bfcol_40` * `bfcol_37`
    ) AS `bfcol_86`
  FROM `bfcte_14`
  WHERE
    REGEXP_CONTAINS(`bfcol_36`, 'green')
), `bfcte_16` AS (
  SELECT
    `bfcol_84`,
    `bfcol_85`,
    COALESCE(SUM(`bfcol_86`), 0) AS `bfcol_90`
  FROM `bfcte_15`
  WHERE
    NOT `bfcol_84` IS NULL AND NOT `bfcol_85` IS NULL
  GROUP BY
    `bfcol_84`,
    `bfcol_85`
)
SELECT
  `bfcol_84` AS `NATION`,
  `bfcol_85` AS `O_YEAR`,
  ROUND(`bfcol_90`, 2) AS `SUM_PROFIT`
FROM `bfcte_16`
ORDER BY
  `bfcol_84` ASC NULLS LAST,
  `bfcol_85` DESC,
  `bfcol_84` ASC NULLS LAST,
  `bfcol_85` ASC NULLS LAST