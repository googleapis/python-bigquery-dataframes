WITH `bfcte_1` AS (
  SELECT
    `P_PARTKEY` AS `bfcol_0`,
    `P_BRAND` AS `bfcol_1`,
    `P_TYPE` AS `bfcol_2`,
    `P_SIZE` AS `bfcol_3`
  FROM `bigframes-dev`.`tpch`.`PART` AS `bft_2` FOR SYSTEM_TIME AS OF '2026-03-10T18:00:00'
), `bfcte_2` AS (
  SELECT
    `PS_PARTKEY` AS `bfcol_4`,
    `PS_SUPPKEY` AS `bfcol_5`
  FROM `bigframes-dev`.`tpch`.`PARTSUPP` AS `bft_1` FOR SYSTEM_TIME AS OF '2026-03-10T18:00:00'
), `bfcte_4` AS (
  SELECT
    *
  FROM `bfcte_1`
  INNER JOIN `bfcte_2`
    ON COALESCE(`bfcol_0`, 0) = COALESCE(`bfcol_4`, 0)
    AND COALESCE(`bfcol_0`, 1) = COALESCE(`bfcol_4`, 1)
), `bfcte_5` AS (
  SELECT
    `bfcol_1` AS `bfcol_48`,
    `bfcol_2` AS `bfcol_49`,
    `bfcol_3` AS `bfcol_50`,
    `bfcol_5` AS `bfcol_51`
  FROM `bfcte_4`
  WHERE
    `bfcol_1` <> 'Brand#45'
    AND NOT (
      REGEXP_CONTAINS(`bfcol_2`, 'MEDIUM POLISHED')
    )
    AND COALESCE(COALESCE(`bfcol_3` IN (49, 14, 23, 45, 19, 3, 36, 9), FALSE), FALSE)
), `bfcte_0` AS (
  SELECT
    `S_SUPPKEY`,
    `S_COMMENT`,
    `S_SUPPKEY` AS `bfcol_54`,
    NOT (
      REGEXP_CONTAINS(`S_COMMENT`, 'Customer.*Complaints')
    ) AS `bfcol_55`
  FROM `bigframes-dev`.`tpch`.`SUPPLIER` AS `bft_0` FOR SYSTEM_TIME AS OF '2026-03-10T18:00:00'
  WHERE
    NOT (
      REGEXP_CONTAINS(`S_COMMENT`, 'Customer.*Complaints')
    )
), `bfcte_3` AS (
  SELECT
    `bfcol_54`
  FROM `bfcte_0`
  GROUP BY
    `bfcol_54`
), `bfcte_6` AS (
  SELECT
    `bfcte_5`.*,
    EXISTS(
      SELECT
        1
      FROM (
        SELECT
          `bfcol_54` AS `bfcol_58`
        FROM `bfcte_3`
      ) AS `bft_3`
      WHERE
        COALESCE(`bfcte_5`.`bfcol_51`, 0) = COALESCE(`bft_3`.`bfcol_58`, 0)
        AND COALESCE(`bfcte_5`.`bfcol_51`, 1) = COALESCE(`bft_3`.`bfcol_58`, 1)
    ) AS `bfcol_59`
  FROM `bfcte_5`
), `bfcte_7` AS (
  SELECT
    *
  FROM `bfcte_6`
  WHERE
    `bfcol_59`
), `bfcte_8` AS (
  SELECT
    `bfcol_48`,
    `bfcol_49`,
    `bfcol_50`,
    COUNT(DISTINCT `bfcol_51`) AS `bfcol_69`
  FROM `bfcte_7`
  WHERE
    NOT `bfcol_48` IS NULL AND NOT `bfcol_49` IS NULL AND NOT `bfcol_50` IS NULL
  GROUP BY
    `bfcol_48`,
    `bfcol_49`,
    `bfcol_50`
)
SELECT
  `bfcol_48` AS `P_BRAND`,
  `bfcol_49` AS `P_TYPE`,
  `bfcol_50` AS `P_SIZE`,
  `bfcol_69` AS `SUPPLIER_CNT`
FROM `bfcte_8`
ORDER BY
  `bfcol_69` DESC,
  `bfcol_48` ASC NULLS LAST,
  `bfcol_49` ASC NULLS LAST,
  `bfcol_50` ASC NULLS LAST