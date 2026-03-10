WITH `bfcte_0` AS (
  SELECT
    `C_CUSTKEY` AS `bfcol_0`,
    `C_NAME` AS `bfcol_1`,
    `C_ADDRESS` AS `bfcol_2`,
    `C_NATIONKEY` AS `bfcol_3`,
    `C_PHONE` AS `bfcol_4`,
    `C_ACCTBAL` AS `bfcol_5`,
    `C_COMMENT` AS `bfcol_6`
  FROM `bigframes-dev`.`tpch`.`CUSTOMER` AS `bft_3` FOR SYSTEM_TIME AS OF '2026-03-10T18:00:00'
), `bfcte_1` AS (
  SELECT
    `O_ORDERKEY` AS `bfcol_7`,
    `O_CUSTKEY` AS `bfcol_8`,
    `O_ORDERDATE` AS `bfcol_9`
  FROM `bigframes-dev`.`tpch`.`ORDERS` AS `bft_2` FOR SYSTEM_TIME AS OF '2026-03-10T18:00:00'
), `bfcte_2` AS (
  SELECT
    *
  FROM `bfcte_0`
  INNER JOIN `bfcte_1`
    ON COALESCE(`bfcol_0`, 0) = COALESCE(`bfcol_8`, 0)
    AND COALESCE(`bfcol_0`, 1) = COALESCE(`bfcol_8`, 1)
), `bfcte_3` AS (
  SELECT
    `bfcol_0` AS `bfcol_10`,
    `bfcol_1` AS `bfcol_11`,
    `bfcol_2` AS `bfcol_12`,
    `bfcol_3` AS `bfcol_13`,
    `bfcol_4` AS `bfcol_14`,
    `bfcol_5` AS `bfcol_15`,
    `bfcol_6` AS `bfcol_16`,
    `bfcol_7` AS `bfcol_17`,
    `bfcol_9` AS `bfcol_18`
  FROM `bfcte_2`
), `bfcte_4` AS (
  SELECT
    `L_ORDERKEY` AS `bfcol_19`,
    `L_EXTENDEDPRICE` AS `bfcol_20`,
    `L_DISCOUNT` AS `bfcol_21`,
    `L_RETURNFLAG` AS `bfcol_22`
  FROM `bigframes-dev`.`tpch`.`LINEITEM` AS `bft_1` FOR SYSTEM_TIME AS OF '2026-03-10T18:00:00'
), `bfcte_5` AS (
  SELECT
    *
  FROM `bfcte_3`
  INNER JOIN `bfcte_4`
    ON COALESCE(`bfcol_17`, 0) = COALESCE(`bfcol_19`, 0)
    AND COALESCE(`bfcol_17`, 1) = COALESCE(`bfcol_19`, 1)
), `bfcte_6` AS (
  SELECT
    `bfcol_10` AS `bfcol_23`,
    `bfcol_11` AS `bfcol_24`,
    `bfcol_12` AS `bfcol_25`,
    `bfcol_13` AS `bfcol_26`,
    `bfcol_14` AS `bfcol_27`,
    `bfcol_15` AS `bfcol_28`,
    `bfcol_16` AS `bfcol_29`,
    `bfcol_18` AS `bfcol_30`,
    `bfcol_20` AS `bfcol_31`,
    `bfcol_21` AS `bfcol_32`,
    `bfcol_22` AS `bfcol_33`
  FROM `bfcte_5`
), `bfcte_7` AS (
  SELECT
    `N_NATIONKEY` AS `bfcol_34`,
    `N_NAME` AS `bfcol_35`
  FROM `bigframes-dev`.`tpch`.`NATION` AS `bft_0` FOR SYSTEM_TIME AS OF '2026-03-10T18:00:00'
), `bfcte_8` AS (
  SELECT
    *
  FROM `bfcte_6`
  INNER JOIN `bfcte_7`
    ON COALESCE(`bfcol_26`, 0) = COALESCE(`bfcol_34`, 0)
    AND COALESCE(`bfcol_26`, 1) = COALESCE(`bfcol_34`, 1)
), `bfcte_9` AS (
  SELECT
    `bfcol_23`,
    `bfcol_24`,
    `bfcol_25`,
    `bfcol_26`,
    `bfcol_27`,
    `bfcol_28`,
    `bfcol_29`,
    `bfcol_30`,
    `bfcol_31`,
    `bfcol_32`,
    `bfcol_33`,
    `bfcol_34`,
    `bfcol_35`,
    `bfcol_23` AS `bfcol_47`,
    `bfcol_24` AS `bfcol_48`,
    `bfcol_25` AS `bfcol_49`,
    `bfcol_27` AS `bfcol_50`,
    `bfcol_28` AS `bfcol_51`,
    `bfcol_29` AS `bfcol_52`,
    `bfcol_31` AS `bfcol_53`,
    `bfcol_32` AS `bfcol_54`,
    `bfcol_35` AS `bfcol_55`,
    (
      (
        `bfcol_30` >= CAST('1993-10-01' AS DATE)
      )
      AND (
        `bfcol_30` < CAST('1994-01-01' AS DATE)
      )
    )
    AND (
      `bfcol_33` = 'R'
    ) AS `bfcol_56`,
    `bfcol_23` AS `bfcol_76`,
    `bfcol_24` AS `bfcol_77`,
    `bfcol_25` AS `bfcol_78`,
    `bfcol_27` AS `bfcol_79`,
    `bfcol_28` AS `bfcol_80`,
    `bfcol_29` AS `bfcol_81`,
    `bfcol_35` AS `bfcol_82`,
    ROUND((
      `bfcol_31` * (
        1 - `bfcol_32`
      )
    ), 2) AS `bfcol_83`
  FROM `bfcte_8`
  WHERE
    (
      (
        `bfcol_30` >= CAST('1993-10-01' AS DATE)
      )
      AND (
        `bfcol_30` < CAST('1994-01-01' AS DATE)
      )
    )
    AND (
      `bfcol_33` = 'R'
    )
), `bfcte_10` AS (
  SELECT
    `bfcol_76`,
    `bfcol_77`,
    `bfcol_80`,
    `bfcol_79`,
    `bfcol_82`,
    `bfcol_78`,
    `bfcol_81`,
    COALESCE(SUM(`bfcol_83`), 0) AS `bfcol_92`
  FROM `bfcte_9`
  WHERE
    NOT `bfcol_76` IS NULL
    AND NOT `bfcol_77` IS NULL
    AND NOT `bfcol_80` IS NULL
    AND NOT `bfcol_79` IS NULL
    AND NOT `bfcol_82` IS NULL
    AND NOT `bfcol_78` IS NULL
    AND NOT `bfcol_81` IS NULL
  GROUP BY
    `bfcol_76`,
    `bfcol_77`,
    `bfcol_80`,
    `bfcol_79`,
    `bfcol_82`,
    `bfcol_78`,
    `bfcol_81`
)
SELECT
  `bfcol_76` AS `C_CUSTKEY`,
  `bfcol_77` AS `C_NAME`,
  `bfcol_92` AS `REVENUE`,
  `bfcol_80` AS `C_ACCTBAL`,
  `bfcol_82` AS `N_NAME`,
  `bfcol_78` AS `C_ADDRESS`,
  `bfcol_79` AS `C_PHONE`,
  `bfcol_81` AS `C_COMMENT`
FROM `bfcte_10`
ORDER BY
  `bfcol_92` DESC,
  `bfcol_76` ASC NULLS LAST,
  `bfcol_77` ASC NULLS LAST,
  `bfcol_80` ASC NULLS LAST,
  `bfcol_79` ASC NULLS LAST,
  `bfcol_82` ASC NULLS LAST,
  `bfcol_78` ASC NULLS LAST,
  `bfcol_81` ASC NULLS LAST
LIMIT 20