WITH `bfcte_0` AS (
  SELECT
    `C_CUSTKEY` AS `bfcol_0`
  FROM `bigframes-dev`.`tpch`.`CUSTOMER` AS `bft_1` FOR SYSTEM_TIME AS OF '2026-03-10T18:00:00'
), `bfcte_1` AS (
  SELECT
    `O_ORDERKEY` AS `bfcol_10`,
    `O_CUSTKEY` AS `bfcol_11`
  FROM `bigframes-dev`.`tpch`.`ORDERS` AS `bft_0` FOR SYSTEM_TIME AS OF '2026-03-10T18:00:00'
  WHERE
    NOT (
      REGEXP_CONTAINS(`O_COMMENT`, 'special.*requests')
    )
), `bfcte_2` AS (
  SELECT
    *
  FROM `bfcte_0`
  LEFT JOIN `bfcte_1`
    ON COALESCE(`bfcol_0`, 0) = COALESCE(`bfcol_11`, 0)
    AND COALESCE(`bfcol_0`, 1) = COALESCE(`bfcol_11`, 1)
), `bfcte_3` AS (
  SELECT
    `bfcol_0`,
    COUNT(`bfcol_10`) AS `bfcol_14`
  FROM `bfcte_2`
  WHERE
    NOT `bfcol_0` IS NULL
  GROUP BY
    `bfcol_0`
), `bfcte_4` AS (
  SELECT
    `bfcol_14`,
    COUNT(1) AS `bfcol_16`
  FROM `bfcte_3`
  WHERE
    NOT `bfcol_14` IS NULL
  GROUP BY
    `bfcol_14`
)
SELECT
  `bfcol_14` AS `C_COUNT`,
  `bfcol_16` AS `CUSTDIST`
FROM `bfcte_4`
ORDER BY
  `bfcol_16` DESC,
  `bfcol_14` DESC