WITH `bfcte_0` AS (
  SELECT
    `P_PARTKEY` AS `bfcol_0`,
    `P_TYPE` AS `bfcol_1`
  FROM `bigframes-dev`.`tpch`.`PART` AS `bft_6` FOR SYSTEM_TIME AS OF '2026-03-10T18:00:00'
), `bfcte_1` AS (
  SELECT
    `L_ORDERKEY` AS `bfcol_2`,
    `L_PARTKEY` AS `bfcol_3`,
    `L_SUPPKEY` AS `bfcol_4`,
    `L_EXTENDEDPRICE` AS `bfcol_5`,
    `L_DISCOUNT` AS `bfcol_6`
  FROM `bigframes-dev`.`tpch`.`LINEITEM` AS `bft_5` FOR SYSTEM_TIME AS OF '2026-03-10T18:00:00'
), `bfcte_2` AS (
  SELECT
    *
  FROM `bfcte_0`
  INNER JOIN `bfcte_1`
    ON COALESCE(`bfcol_0`, 0) = COALESCE(`bfcol_3`, 0)
    AND COALESCE(`bfcol_0`, 1) = COALESCE(`bfcol_3`, 1)
), `bfcte_3` AS (
  SELECT
    `bfcol_1` AS `bfcol_7`,
    `bfcol_2` AS `bfcol_8`,
    `bfcol_4` AS `bfcol_9`,
    `bfcol_5` AS `bfcol_10`,
    `bfcol_6` AS `bfcol_11`
  FROM `bfcte_2`
), `bfcte_4` AS (
  SELECT
    `S_SUPPKEY` AS `bfcol_12`,
    `S_NATIONKEY` AS `bfcol_13`
  FROM `bigframes-dev`.`tpch`.`SUPPLIER` AS `bft_4` FOR SYSTEM_TIME AS OF '2026-03-10T18:00:00'
), `bfcte_5` AS (
  SELECT
    *
  FROM `bfcte_3`
  INNER JOIN `bfcte_4`
    ON COALESCE(`bfcol_9`, 0) = COALESCE(`bfcol_12`, 0)
    AND COALESCE(`bfcol_9`, 1) = COALESCE(`bfcol_12`, 1)
), `bfcte_6` AS (
  SELECT
    `bfcol_7` AS `bfcol_14`,
    `bfcol_8` AS `bfcol_15`,
    `bfcol_10` AS `bfcol_16`,
    `bfcol_11` AS `bfcol_17`,
    `bfcol_13` AS `bfcol_18`
  FROM `bfcte_5`
), `bfcte_7` AS (
  SELECT
    `O_ORDERKEY` AS `bfcol_19`,
    `O_CUSTKEY` AS `bfcol_20`,
    `O_ORDERDATE` AS `bfcol_21`
  FROM `bigframes-dev`.`tpch`.`ORDERS` AS `bft_3` FOR SYSTEM_TIME AS OF '2026-03-10T18:00:00'
), `bfcte_8` AS (
  SELECT
    *
  FROM `bfcte_6`
  INNER JOIN `bfcte_7`
    ON COALESCE(`bfcol_15`, 0) = COALESCE(`bfcol_19`, 0)
    AND COALESCE(`bfcol_15`, 1) = COALESCE(`bfcol_19`, 1)
), `bfcte_9` AS (
  SELECT
    `bfcol_14` AS `bfcol_22`,
    `bfcol_16` AS `bfcol_23`,
    `bfcol_17` AS `bfcol_24`,
    `bfcol_18` AS `bfcol_25`,
    `bfcol_20` AS `bfcol_26`,
    `bfcol_21` AS `bfcol_27`
  FROM `bfcte_8`
), `bfcte_10` AS (
  SELECT
    `C_CUSTKEY` AS `bfcol_28`,
    `C_NATIONKEY` AS `bfcol_29`
  FROM `bigframes-dev`.`tpch`.`CUSTOMER` AS `bft_2` FOR SYSTEM_TIME AS OF '2026-03-10T18:00:00'
), `bfcte_11` AS (
  SELECT
    *
  FROM `bfcte_9`
  INNER JOIN `bfcte_10`
    ON COALESCE(`bfcol_26`, 0) = COALESCE(`bfcol_28`, 0)
    AND COALESCE(`bfcol_26`, 1) = COALESCE(`bfcol_28`, 1)
), `bfcte_12` AS (
  SELECT
    `bfcol_22` AS `bfcol_30`,
    `bfcol_23` AS `bfcol_31`,
    `bfcol_24` AS `bfcol_32`,
    `bfcol_25` AS `bfcol_33`,
    `bfcol_27` AS `bfcol_34`,
    `bfcol_29` AS `bfcol_35`
  FROM `bfcte_11`
), `bfcte_13` AS (
  SELECT
    `N_NATIONKEY` AS `bfcol_36`,
    `N_REGIONKEY` AS `bfcol_37`
  FROM `bigframes-dev`.`tpch`.`NATION` AS `bft_0` FOR SYSTEM_TIME AS OF '2026-03-10T18:00:00'
), `bfcte_14` AS (
  SELECT
    *
  FROM `bfcte_12`
  INNER JOIN `bfcte_13`
    ON COALESCE(`bfcol_35`, 0) = COALESCE(`bfcol_36`, 0)
    AND COALESCE(`bfcol_35`, 1) = COALESCE(`bfcol_36`, 1)
), `bfcte_15` AS (
  SELECT
    `bfcol_30` AS `bfcol_38`,
    `bfcol_31` AS `bfcol_39`,
    `bfcol_32` AS `bfcol_40`,
    `bfcol_33` AS `bfcol_41`,
    `bfcol_34` AS `bfcol_42`,
    `bfcol_37` AS `bfcol_43`
  FROM `bfcte_14`
), `bfcte_16` AS (
  SELECT
    `R_REGIONKEY` AS `bfcol_44`,
    `R_NAME` AS `bfcol_45`
  FROM `bigframes-dev`.`tpch`.`REGION` AS `bft_1` FOR SYSTEM_TIME AS OF '2026-03-10T18:00:00'
), `bfcte_17` AS (
  SELECT
    *
  FROM `bfcte_15`
  INNER JOIN `bfcte_16`
    ON COALESCE(`bfcol_43`, 0) = COALESCE(`bfcol_44`, 0)
    AND COALESCE(`bfcol_43`, 1) = COALESCE(`bfcol_44`, 1)
), `bfcte_18` AS (
  SELECT
    `bfcol_38` AS `bfcol_64`,
    `bfcol_39` AS `bfcol_65`,
    `bfcol_40` AS `bfcol_66`,
    `bfcol_41` AS `bfcol_67`,
    `bfcol_42` AS `bfcol_68`
  FROM `bfcte_17`
  WHERE
    `bfcol_45` = 'AMERICA'
), `bfcte_19` AS (
  SELECT
    `N_NATIONKEY` AS `bfcol_69`,
    `N_NAME` AS `bfcol_70`
  FROM `bigframes-dev`.`tpch`.`NATION` AS `bft_0` FOR SYSTEM_TIME AS OF '2026-03-10T18:00:00'
), `bfcte_20` AS (
  SELECT
    *
  FROM `bfcte_18`
  INNER JOIN `bfcte_19`
    ON COALESCE(`bfcol_67`, 0) = COALESCE(`bfcol_69`, 0)
    AND COALESCE(`bfcol_67`, 1) = COALESCE(`bfcol_69`, 1)
), `bfcte_21` AS (
  SELECT
    `bfcol_64`,
    `bfcol_65`,
    `bfcol_66`,
    `bfcol_67`,
    `bfcol_68`,
    `bfcol_69`,
    `bfcol_70`,
    `bfcol_64` AS `bfcol_76`,
    `bfcol_65` AS `bfcol_77`,
    `bfcol_66` AS `bfcol_78`,
    `bfcol_68` AS `bfcol_79`,
    `bfcol_70` AS `bfcol_80`,
    (
      `bfcol_68` >= CAST('1995-01-01' AS DATE)
    )
    AND (
      `bfcol_68` <= CAST('1996-12-31' AS DATE)
    ) AS `bfcol_81`,
    `bfcol_65` AS `bfcol_93`,
    `bfcol_66` AS `bfcol_94`,
    `bfcol_68` AS `bfcol_95`,
    `bfcol_70` AS `bfcol_96`,
    `bfcol_64` = 'ECONOMY ANODIZED STEEL' AS `bfcol_97`,
    `bfcol_65` AS `bfcol_107`,
    `bfcol_66` AS `bfcol_108`,
    `bfcol_70` AS `bfcol_109`,
    EXTRACT(YEAR FROM `bfcol_68`) AS `bfcol_110`,
    `bfcol_70` AS `bfcol_115`,
    EXTRACT(YEAR FROM `bfcol_68`) AS `bfcol_116`,
    `bfcol_65` * (
      1.0 - `bfcol_66`
    ) AS `bfcol_117`,
    EXTRACT(YEAR FROM `bfcol_68`) AS `bfcol_121`,
    `bfcol_65` * (
      1.0 - `bfcol_66`
    ) AS `bfcol_122`,
    IF(`bfcol_70` = 'BRAZIL', `bfcol_65` * (
      1.0 - `bfcol_66`
    ), 0) AS `bfcol_123`,
    EXTRACT(YEAR FROM `bfcol_68`) AS `bfcol_127`,
    IF(`bfcol_70` = 'BRAZIL', `bfcol_65` * (
      1.0 - `bfcol_66`
    ), 0) AS `bfcol_128`,
    `bfcol_65` * (
      1.0 - `bfcol_66`
    ) AS `bfcol_129`
  FROM `bfcte_20`
  WHERE
    (
      `bfcol_68` >= CAST('1995-01-01' AS DATE)
    )
    AND (
      `bfcol_68` <= CAST('1996-12-31' AS DATE)
    )
    AND `bfcol_64` = 'ECONOMY ANODIZED STEEL'
), `bfcte_22` AS (
  SELECT
    `bfcol_127`,
    COALESCE(SUM(`bfcol_128`), 0) AS `bfcol_133`,
    COALESCE(SUM(`bfcol_129`), 0) AS `bfcol_134`
  FROM `bfcte_21`
  WHERE
    NOT `bfcol_127` IS NULL
  GROUP BY
    `bfcol_127`
)
SELECT
  `bfcol_127` AS `O_YEAR`,
  ROUND(IEEE_DIVIDE(`bfcol_133`, `bfcol_134`), 2) AS `MKT_SHARE`
FROM `bfcte_22`
ORDER BY
  `bfcol_127` ASC NULLS LAST,
  `bfcol_127` ASC NULLS LAST