WITH `bfcte_4` AS (
  SELECT
    `O_ORDERKEY` AS `bfcol_0`,
    `O_CUSTKEY` AS `bfcol_1`,
    `O_TOTALPRICE` AS `bfcol_2`,
    `O_ORDERDATE` AS `bfcol_3`
  FROM `bigframes-dev`.`tpch`.`ORDERS` AS `bft_2` FOR SYSTEM_TIME AS OF '2026-03-10T18:00:00'
), `bfcte_0` AS (
  SELECT
    `L_ORDERKEY`,
    `L_QUANTITY`
  FROM `bigframes-dev`.`tpch`.`LINEITEM` AS `bft_1` FOR SYSTEM_TIME AS OF '2026-03-10T18:00:00'
), `bfcte_1` AS (
  SELECT
    `L_ORDERKEY`,
    COALESCE(SUM(`L_QUANTITY`), 0) AS `bfcol_6`
  FROM `bfcte_0`
  WHERE
    NOT `L_ORDERKEY` IS NULL
  GROUP BY
    `L_ORDERKEY`
), `bfcte_2` AS (
  SELECT
    `L_ORDERKEY`,
    `bfcol_6`,
    `L_ORDERKEY` AS `bfcol_7`,
    `bfcol_6` > 300 AS `bfcol_8`
  FROM `bfcte_1`
  WHERE
    `bfcol_6` > 300
), `bfcte_3` AS (
  SELECT
    `bfcol_7`
  FROM `bfcte_2`
  GROUP BY
    `bfcol_7`
), `bfcte_5` AS (
  SELECT
    `bfcte_4`.*,
    EXISTS(
      SELECT
        1
      FROM (
        SELECT
          `bfcol_7` AS `bfcol_11`
        FROM `bfcte_3`
      ) AS `bft_3`
      WHERE
        COALESCE(`bfcte_4`.`bfcol_0`, 0) = COALESCE(`bft_3`.`bfcol_11`, 0)
        AND COALESCE(`bfcte_4`.`bfcol_0`, 1) = COALESCE(`bft_3`.`bfcol_11`, 1)
    ) AS `bfcol_12`
  FROM `bfcte_4`
), `bfcte_6` AS (
  SELECT
    `bfcol_0` AS `bfcol_18`,
    `bfcol_1` AS `bfcol_19`,
    `bfcol_2` AS `bfcol_20`,
    `bfcol_3` AS `bfcol_21`
  FROM `bfcte_5`
  WHERE
    `bfcol_12`
), `bfcte_7` AS (
  SELECT
    `L_ORDERKEY` AS `bfcol_22`,
    `L_QUANTITY` AS `bfcol_23`
  FROM `bigframes-dev`.`tpch`.`LINEITEM` AS `bft_1` FOR SYSTEM_TIME AS OF '2026-03-10T18:00:00'
), `bfcte_8` AS (
  SELECT
    *
  FROM `bfcte_6`
  INNER JOIN `bfcte_7`
    ON COALESCE(`bfcol_18`, 0) = COALESCE(`bfcol_22`, 0)
    AND COALESCE(`bfcol_18`, 1) = COALESCE(`bfcol_22`, 1)
), `bfcte_9` AS (
  SELECT
    `bfcol_18` AS `bfcol_24`,
    `bfcol_19` AS `bfcol_25`,
    `bfcol_20` AS `bfcol_26`,
    `bfcol_21` AS `bfcol_27`,
    `bfcol_23` AS `bfcol_28`
  FROM `bfcte_8`
), `bfcte_10` AS (
  SELECT
    `C_CUSTKEY` AS `bfcol_29`,
    `C_NAME` AS `bfcol_30`
  FROM `bigframes-dev`.`tpch`.`CUSTOMER` AS `bft_0` FOR SYSTEM_TIME AS OF '2026-03-10T18:00:00'
), `bfcte_11` AS (
  SELECT
    *
  FROM `bfcte_9`
  INNER JOIN `bfcte_10`
    ON COALESCE(`bfcol_25`, 0) = COALESCE(`bfcol_29`, 0)
    AND COALESCE(`bfcol_25`, 1) = COALESCE(`bfcol_29`, 1)
), `bfcte_12` AS (
  SELECT
    `bfcol_30`,
    `bfcol_29`,
    `bfcol_24`,
    `bfcol_27`,
    `bfcol_26`,
    COALESCE(SUM(`bfcol_28`), 0) AS `bfcol_37`
  FROM `bfcte_11`
  WHERE
    NOT `bfcol_30` IS NULL
    AND NOT `bfcol_29` IS NULL
    AND NOT `bfcol_24` IS NULL
    AND NOT `bfcol_27` IS NULL
    AND NOT `bfcol_26` IS NULL
  GROUP BY
    `bfcol_30`,
    `bfcol_29`,
    `bfcol_24`,
    `bfcol_27`,
    `bfcol_26`
)
SELECT
  `bfcol_30` AS `C_NAME`,
  `bfcol_29` AS `C_CUSTKEY`,
  `bfcol_24` AS `O_ORDERKEY`,
  `bfcol_27` AS `O_ORDERDAT`,
  `bfcol_26` AS `O_TOTALPRICE`,
  `bfcol_37` AS `COL6`
FROM `bfcte_12`
ORDER BY
  `bfcol_26` DESC,
  `bfcol_27` ASC NULLS LAST,
  `bfcol_30` ASC NULLS LAST,
  `bfcol_29` ASC NULLS LAST,
  `bfcol_24` ASC NULLS LAST
LIMIT 100