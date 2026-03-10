WITH `bfcte_3` AS (
  SELECT
    `C_CUSTKEY` AS `bfcol_6`
  FROM `bigframes-dev`.`tpch`.`CUSTOMER` AS `bft_2` FOR SYSTEM_TIME AS OF '2026-03-10T18:00:00'
  WHERE
    `C_MKTSEGMENT` = 'BUILDING'
), `bfcte_0` AS (
  SELECT
    `L_ORDERKEY` AS `bfcol_19`,
    `L_EXTENDEDPRICE` AS `bfcol_20`,
    `L_DISCOUNT` AS `bfcol_21`
  FROM `bigframes-dev`.`tpch`.`LINEITEM` AS `bft_1` FOR SYSTEM_TIME AS OF '2026-03-10T18:00:00'
  WHERE
    `L_SHIPDATE` > CAST('1995-03-15' AS DATE)
), `bfcte_1` AS (
  SELECT
    `O_ORDERKEY` AS `bfcol_36`,
    `O_CUSTKEY` AS `bfcol_37`,
    `O_ORDERDATE` AS `bfcol_38`,
    `O_SHIPPRIORITY` AS `bfcol_39`
  FROM `bigframes-dev`.`tpch`.`ORDERS` AS `bft_0` FOR SYSTEM_TIME AS OF '2026-03-10T18:00:00'
  WHERE
    `O_ORDERDATE` < CAST('1995-03-15' AS DATE)
), `bfcte_2` AS (
  SELECT
    *
  FROM `bfcte_0`
  INNER JOIN `bfcte_1`
    ON COALESCE(`bfcol_19`, 0) = COALESCE(`bfcol_36`, 0)
    AND COALESCE(`bfcol_19`, 1) = COALESCE(`bfcol_36`, 1)
), `bfcte_4` AS (
  SELECT
    `bfcol_20` AS `bfcol_40`,
    `bfcol_21` AS `bfcol_41`,
    `bfcol_36` AS `bfcol_42`,
    `bfcol_37` AS `bfcol_43`,
    `bfcol_38` AS `bfcol_44`,
    `bfcol_39` AS `bfcol_45`
  FROM `bfcte_2`
), `bfcte_5` AS (
  SELECT
    *
  FROM `bfcte_3`
  INNER JOIN `bfcte_4`
    ON COALESCE(`bfcol_6`, 0) = COALESCE(`bfcol_43`, 0)
    AND COALESCE(`bfcol_6`, 1) = COALESCE(`bfcol_43`, 1)
), `bfcte_6` AS (
  SELECT
    `bfcol_6`,
    `bfcol_40`,
    `bfcol_41`,
    `bfcol_42`,
    `bfcol_43`,
    `bfcol_44`,
    `bfcol_45`,
    `bfcol_42` AS `bfcol_51`,
    `bfcol_44` AS `bfcol_52`,
    `bfcol_45` AS `bfcol_53`,
    `bfcol_40` * (
      1 - `bfcol_41`
    ) AS `bfcol_54`
  FROM `bfcte_5`
), `bfcte_7` AS (
  SELECT
    `bfcol_51`,
    `bfcol_52`,
    `bfcol_53`,
    COALESCE(SUM(`bfcol_54`), 0) AS `bfcol_59`
  FROM `bfcte_6`
  WHERE
    NOT `bfcol_51` IS NULL AND NOT `bfcol_52` IS NULL AND NOT `bfcol_53` IS NULL
  GROUP BY
    `bfcol_51`,
    `bfcol_52`,
    `bfcol_53`
)
SELECT
  `bfcol_51` AS `L_ORDERKEY`,
  `bfcol_59` AS `REVENUE`,
  `bfcol_52` AS `O_ORDERDATE`,
  `bfcol_53` AS `O_SHIPPRIORITY`
FROM `bfcte_7`
ORDER BY
  `bfcol_59` DESC,
  `bfcol_52` ASC NULLS LAST,
  `bfcol_51` ASC NULLS LAST,
  `bfcol_53` ASC NULLS LAST
LIMIT 10