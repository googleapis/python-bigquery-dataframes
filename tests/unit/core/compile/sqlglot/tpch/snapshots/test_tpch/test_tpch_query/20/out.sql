WITH `bfcte_2` AS (
  SELECT
    `S_SUPPKEY` AS `bfcol_0`,
    `S_NAME` AS `bfcol_1`,
    `S_ADDRESS` AS `bfcol_2`,
    `S_NATIONKEY` AS `bfcol_3`
  FROM `bigframes-dev`.`tpch`.`SUPPLIER` AS `bft_4` FOR SYSTEM_TIME AS OF '2026-03-10T18:00:00'
), `bfcte_3` AS (
  SELECT
    `N_NATIONKEY` AS `bfcol_10`
  FROM `bigframes-dev`.`tpch`.`NATION` AS `bft_3` FOR SYSTEM_TIME AS OF '2026-03-10T18:00:00'
  WHERE
    `N_NAME` = 'CANADA'
), `bfcte_6` AS (
  SELECT
    *
  FROM `bfcte_2`
  INNER JOIN `bfcte_3`
    ON COALESCE(`bfcol_3`, 0) = COALESCE(`bfcol_10`, 0)
    AND COALESCE(`bfcol_3`, 1) = COALESCE(`bfcol_10`, 1)
), `bfcte_14` AS (
  SELECT
    `bfcol_0` AS `bfcol_11`,
    `bfcol_1` AS `bfcol_12`,
    `bfcol_2` AS `bfcol_13`
  FROM `bfcte_6`
), `bfcte_1` AS (
  SELECT
    `L_PARTKEY`,
    `L_SUPPKEY`,
    `L_QUANTITY`,
    `L_SHIPDATE`,
    `L_PARTKEY` AS `bfcol_18`,
    `L_SUPPKEY` AS `bfcol_19`,
    `L_QUANTITY` AS `bfcol_20`,
    (
      `L_SHIPDATE` >= CAST('1994-01-01' AS DATE)
    )
    AND (
      `L_SHIPDATE` < CAST('1995-01-01' AS DATE)
    ) AS `bfcol_21`
  FROM `bigframes-dev`.`tpch`.`LINEITEM` AS `bft_2` FOR SYSTEM_TIME AS OF '2026-03-10T18:00:00'
  WHERE
    (
      `L_SHIPDATE` >= CAST('1994-01-01' AS DATE)
    )
    AND (
      `L_SHIPDATE` < CAST('1995-01-01' AS DATE)
    )
), `bfcte_5` AS (
  SELECT
    `bfcol_18`,
    `bfcol_19`,
    COALESCE(SUM(`bfcol_20`), 0) AS `bfcol_29`
  FROM `bfcte_1`
  WHERE
    NOT `bfcol_18` IS NULL AND NOT `bfcol_19` IS NULL
  GROUP BY
    `bfcol_18`,
    `bfcol_19`
), `bfcte_9` AS (
  SELECT
    `bfcol_18` AS `bfcol_33`,
    `bfcol_19` AS `bfcol_34`,
    `bfcol_29` * 0.5 AS `bfcol_35`
  FROM `bfcte_5`
), `bfcte_7` AS (
  SELECT
    `PS_PARTKEY` AS `bfcol_36`,
    `PS_SUPPKEY` AS `bfcol_37`,
    `PS_AVAILQTY` AS `bfcol_38`
  FROM `bigframes-dev`.`tpch`.`PARTSUPP` AS `bft_1` FOR SYSTEM_TIME AS OF '2026-03-10T18:00:00'
), `bfcte_0` AS (
  SELECT
    `P_PARTKEY`,
    `P_NAME`,
    `P_PARTKEY` AS `bfcol_41`,
    STARTS_WITH(`P_NAME`, 'forest') AS `bfcol_42`
  FROM `bigframes-dev`.`tpch`.`PART` AS `bft_0` FOR SYSTEM_TIME AS OF '2026-03-10T18:00:00'
  WHERE
    STARTS_WITH(`P_NAME`, 'forest')
), `bfcte_4` AS (
  SELECT
    `bfcol_41`
  FROM `bfcte_0`
  GROUP BY
    `bfcol_41`
), `bfcte_8` AS (
  SELECT
    `bfcte_7`.*,
    EXISTS(
      SELECT
        1
      FROM (
        SELECT
          `bfcol_41` AS `bfcol_45`
        FROM `bfcte_4`
      ) AS `bft_5`
      WHERE
        COALESCE(`bfcte_7`.`bfcol_36`, 0) = COALESCE(`bft_5`.`bfcol_45`, 0)
        AND COALESCE(`bfcte_7`.`bfcol_36`, 1) = COALESCE(`bft_5`.`bfcol_45`, 1)
    ) AS `bfcol_46`
  FROM `bfcte_7`
), `bfcte_10` AS (
  SELECT
    `bfcol_36` AS `bfcol_51`,
    `bfcol_37` AS `bfcol_52`,
    `bfcol_38` AS `bfcol_53`
  FROM `bfcte_8`
  WHERE
    `bfcol_46`
), `bfcte_11` AS (
  SELECT
    *
  FROM `bfcte_9`
  INNER JOIN `bfcte_10`
    ON `bfcol_34` = `bfcol_52` AND `bfcol_33` = `bfcol_51`
), `bfcte_12` AS (
  SELECT
    `bfcol_33`,
    `bfcol_34`,
    `bfcol_35`,
    `bfcol_51`,
    `bfcol_52`,
    `bfcol_53`,
    `bfcol_52` AS `bfcol_57`,
    `bfcol_53` > `bfcol_35` AS `bfcol_58`
  FROM `bfcte_11`
  WHERE
    `bfcol_53` > `bfcol_35`
), `bfcte_13` AS (
  SELECT
    `bfcol_57`
  FROM `bfcte_12`
  GROUP BY
    `bfcol_57`
), `bfcte_15` AS (
  SELECT
    `bfcte_14`.*,
    EXISTS(
      SELECT
        1
      FROM (
        SELECT
          `bfcol_57` AS `bfcol_61`
        FROM `bfcte_13`
      ) AS `bft_6`
      WHERE
        COALESCE(`bfcte_14`.`bfcol_11`, 0) = COALESCE(`bft_6`.`bfcol_61`, 0)
        AND COALESCE(`bfcte_14`.`bfcol_11`, 1) = COALESCE(`bft_6`.`bfcol_61`, 1)
    ) AS `bfcol_62`
  FROM `bfcte_14`
)
SELECT
  `bfcol_12` AS `S_NAME`,
  `bfcol_13` AS `S_ADDRESS`
FROM `bfcte_15`
WHERE
  `bfcol_62`
ORDER BY
  `bfcol_12` ASC NULLS LAST