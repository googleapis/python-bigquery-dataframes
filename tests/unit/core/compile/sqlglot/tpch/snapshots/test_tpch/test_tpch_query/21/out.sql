WITH `bfcte_1` AS (
  SELECT
    `L_ORDERKEY`
  FROM `bigframes-dev`.`tpch`.`LINEITEM` AS `bft_3` FOR SYSTEM_TIME AS OF '2026-03-10T18:00:00'
), `bfcte_3` AS (
  SELECT
    `L_ORDERKEY`,
    COUNT(1) AS `bfcol_1`
  FROM `bfcte_1`
  WHERE
    NOT `L_ORDERKEY` IS NULL
  GROUP BY
    `L_ORDERKEY`
), `bfcte_6` AS (
  SELECT
    `L_ORDERKEY` AS `bfcol_6`
  FROM `bfcte_3`
  WHERE
    `bfcol_1` > 1
), `bfcte_7` AS (
  SELECT
    `L_ORDERKEY` AS `bfcol_14`
  FROM `bigframes-dev`.`tpch`.`LINEITEM` AS `bft_3` FOR SYSTEM_TIME AS OF '2026-03-10T18:00:00'
  WHERE
    `L_RECEIPTDATE` > `L_COMMITDATE`
), `bfcte_9` AS (
  SELECT
    *
  FROM `bfcte_6`
  INNER JOIN `bfcte_7`
    ON `bfcol_6` = `bfcol_14`
), `bfcte_10` AS (
  SELECT
    `bfcol_6`,
    COUNT(1) AS `bfcol_16`
  FROM `bfcte_9`
  GROUP BY
    `bfcol_6`
), `bfcte_11` AS (
  SELECT
    `bfcol_6` AS `bfcol_15`,
    `bfcol_16`
  FROM `bfcte_10`
), `bfcte_0` AS (
  SELECT
    `L_ORDERKEY`
  FROM `bigframes-dev`.`tpch`.`LINEITEM` AS `bft_3` FOR SYSTEM_TIME AS OF '2026-03-10T18:00:00'
), `bfcte_2` AS (
  SELECT
    `L_ORDERKEY`,
    COUNT(1) AS `bfcol_18`
  FROM `bfcte_0`
  WHERE
    NOT `L_ORDERKEY` IS NULL
  GROUP BY
    `L_ORDERKEY`
), `bfcte_4` AS (
  SELECT
    `L_ORDERKEY` AS `bfcol_23`
  FROM `bfcte_2`
  WHERE
    `bfcol_18` > 1
), `bfcte_5` AS (
  SELECT
    `L_ORDERKEY` AS `bfcol_34`,
    `L_SUPPKEY` AS `bfcol_35`
  FROM `bigframes-dev`.`tpch`.`LINEITEM` AS `bft_3` FOR SYSTEM_TIME AS OF '2026-03-10T18:00:00'
  WHERE
    `L_RECEIPTDATE` > `L_COMMITDATE`
), `bfcte_8` AS (
  SELECT
    *
  FROM `bfcte_4`
  INNER JOIN `bfcte_5`
    ON `bfcol_23` = `bfcol_34`
), `bfcte_12` AS (
  SELECT
    `bfcol_23` AS `bfcol_36`,
    `bfcol_35` AS `bfcol_37`
  FROM `bfcte_8`
), `bfcte_13` AS (
  SELECT
    *
  FROM `bfcte_11`
  INNER JOIN `bfcte_12`
    ON `bfcol_15` = `bfcol_36`
), `bfcte_14` AS (
  SELECT
    `bfcol_15` AS `bfcol_38`,
    `bfcol_16` AS `bfcol_39`,
    `bfcol_37` AS `bfcol_40`
  FROM `bfcte_13`
), `bfcte_15` AS (
  SELECT
    `S_SUPPKEY` AS `bfcol_41`,
    `S_NAME` AS `bfcol_42`,
    `S_NATIONKEY` AS `bfcol_43`
  FROM `bigframes-dev`.`tpch`.`SUPPLIER` AS `bft_2` FOR SYSTEM_TIME AS OF '2026-03-10T18:00:00'
), `bfcte_16` AS (
  SELECT
    *
  FROM `bfcte_14`
  INNER JOIN `bfcte_15`
    ON COALESCE(`bfcol_40`, 0) = COALESCE(`bfcol_41`, 0)
    AND COALESCE(`bfcol_40`, 1) = COALESCE(`bfcol_41`, 1)
), `bfcte_17` AS (
  SELECT
    `bfcol_38` AS `bfcol_44`,
    `bfcol_39` AS `bfcol_45`,
    `bfcol_42` AS `bfcol_46`,
    `bfcol_43` AS `bfcol_47`
  FROM `bfcte_16`
), `bfcte_18` AS (
  SELECT
    `N_NATIONKEY` AS `bfcol_48`,
    `N_NAME` AS `bfcol_49`
  FROM `bigframes-dev`.`tpch`.`NATION` AS `bft_1` FOR SYSTEM_TIME AS OF '2026-03-10T18:00:00'
), `bfcte_19` AS (
  SELECT
    *
  FROM `bfcte_17`
  INNER JOIN `bfcte_18`
    ON COALESCE(`bfcol_47`, 0) = COALESCE(`bfcol_48`, 0)
    AND COALESCE(`bfcol_47`, 1) = COALESCE(`bfcol_48`, 1)
), `bfcte_20` AS (
  SELECT
    `bfcol_44` AS `bfcol_50`,
    `bfcol_45` AS `bfcol_51`,
    `bfcol_46` AS `bfcol_52`,
    `bfcol_49` AS `bfcol_53`
  FROM `bfcte_19`
), `bfcte_21` AS (
  SELECT
    `O_ORDERKEY` AS `bfcol_54`,
    `O_ORDERSTATUS` AS `bfcol_55`
  FROM `bigframes-dev`.`tpch`.`ORDERS` AS `bft_0` FOR SYSTEM_TIME AS OF '2026-03-10T18:00:00'
), `bfcte_22` AS (
  SELECT
    *
  FROM `bfcte_20`
  INNER JOIN `bfcte_21`
    ON `bfcol_50` = `bfcol_54`
), `bfcte_23` AS (
  SELECT
    `bfcol_50`,
    `bfcol_51`,
    `bfcol_52`,
    `bfcol_53`,
    `bfcol_54`,
    `bfcol_55`,
    `bfcol_52` AS `bfcol_60`,
    (
      (
        `bfcol_51` = 1
      ) AND (
        `bfcol_53` = 'SAUDI ARABIA'
      )
    )
    AND (
      `bfcol_55` = 'F'
    ) AS `bfcol_61`
  FROM `bfcte_22`
  WHERE
    (
      (
        `bfcol_51` = 1
      ) AND (
        `bfcol_53` = 'SAUDI ARABIA'
      )
    )
    AND (
      `bfcol_55` = 'F'
    )
), `bfcte_24` AS (
  SELECT
    `bfcol_60`,
    COUNT(1) AS `bfcol_65`
  FROM `bfcte_23`
  WHERE
    NOT `bfcol_60` IS NULL
  GROUP BY
    `bfcol_60`
)
SELECT
  `bfcol_60` AS `S_NAME`,
  `bfcol_65` AS `NUMWAIT`
FROM `bfcte_24`
ORDER BY
  `bfcol_65` DESC,
  `bfcol_60` ASC NULLS LAST
LIMIT 100