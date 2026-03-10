WITH `bfcte_12` AS (
  SELECT
    *
  FROM UNNEST(ARRAY<STRUCT<`bfcol_0` STRING, `bfcol_1` INT64, `bfcol_2` INT64>>[STRUCT('L_EXTENDEDPRICE', 0, 0)])
), `bfcte_2` AS (
  SELECT
    `L_PARTKEY` AS `bfcol_3`,
    `L_QUANTITY` AS `bfcol_4`
  FROM `bigframes-dev`.`tpch`.`LINEITEM` AS `bft_1` FOR SYSTEM_TIME AS OF '2026-03-10T18:00:00'
), `bfcte_3` AS (
  SELECT
    `P_PARTKEY` AS `bfcol_12`
  FROM `bigframes-dev`.`tpch`.`PART` AS `bft_0` FOR SYSTEM_TIME AS OF '2026-03-10T18:00:00'
  WHERE
    (
      `P_BRAND` = 'Brand#23'
    ) AND (
      `P_CONTAINER` = 'MED BOX'
    )
), `bfcte_5` AS (
  SELECT
    *
  FROM `bfcte_2`
  RIGHT JOIN `bfcte_3`
    ON COALESCE(`bfcol_3`, 0) = COALESCE(`bfcol_12`, 0)
    AND COALESCE(`bfcol_3`, 1) = COALESCE(`bfcol_12`, 1)
), `bfcte_6` AS (
  SELECT
    `bfcol_12`,
    AVG(`bfcol_4`) AS `bfcol_15`
  FROM `bfcte_5`
  WHERE
    NOT `bfcol_12` IS NULL
  GROUP BY
    `bfcol_12`
), `bfcte_7` AS (
  SELECT
    `bfcol_12` AS `bfcol_18`,
    `bfcol_15` * 0.2 AS `bfcol_19`
  FROM `bfcte_6`
), `bfcte_0` AS (
  SELECT
    `L_PARTKEY` AS `bfcol_20`,
    `L_QUANTITY` AS `bfcol_21`,
    `L_EXTENDEDPRICE` AS `bfcol_22`
  FROM `bigframes-dev`.`tpch`.`LINEITEM` AS `bft_1` FOR SYSTEM_TIME AS OF '2026-03-10T18:00:00'
), `bfcte_1` AS (
  SELECT
    `P_PARTKEY` AS `bfcol_30`
  FROM `bigframes-dev`.`tpch`.`PART` AS `bft_0` FOR SYSTEM_TIME AS OF '2026-03-10T18:00:00'
  WHERE
    (
      `P_BRAND` = 'Brand#23'
    ) AND (
      `P_CONTAINER` = 'MED BOX'
    )
), `bfcte_4` AS (
  SELECT
    *
  FROM `bfcte_0`
  RIGHT JOIN `bfcte_1`
    ON COALESCE(`bfcol_20`, 0) = COALESCE(`bfcol_30`, 0)
    AND COALESCE(`bfcol_20`, 1) = COALESCE(`bfcol_30`, 1)
), `bfcte_8` AS (
  SELECT
    `bfcol_21` AS `bfcol_31`,
    `bfcol_22` AS `bfcol_32`,
    `bfcol_30` AS `bfcol_33`
  FROM `bfcte_4`
), `bfcte_9` AS (
  SELECT
    *
  FROM `bfcte_7`
  INNER JOIN `bfcte_8`
    ON `bfcol_18` = `bfcol_33`
), `bfcte_10` AS (
  SELECT
    `bfcol_18`,
    `bfcol_19`,
    `bfcol_31`,
    `bfcol_32`,
    `bfcol_33`,
    `bfcol_32` AS `bfcol_37`,
    `bfcol_31` < `bfcol_19` AS `bfcol_38`
  FROM `bfcte_9`
  WHERE
    `bfcol_31` < `bfcol_19`
), `bfcte_11` AS (
  SELECT
    COALESCE(SUM(`bfcol_37`), 0) AS `bfcol_42`
  FROM `bfcte_10`
), `bfcte_13` AS (
  SELECT
    `bfcol_42`,
    0 AS `bfcol_43`
  FROM `bfcte_11`
), `bfcte_14` AS (
  SELECT
    *
  FROM `bfcte_12`
  CROSS JOIN `bfcte_13`
), `bfcte_15` AS (
  SELECT
    `bfcol_0`,
    `bfcol_1`,
    `bfcol_2`,
    `bfcol_42`,
    `bfcol_43`,
    CASE WHEN `bfcol_2` = 0 THEN `bfcol_42` END AS `bfcol_44`,
    IF(`bfcol_43` = 0, CASE WHEN `bfcol_2` = 0 THEN `bfcol_42` END, NULL) AS `bfcol_49`
  FROM `bfcte_14`
), `bfcte_16` AS (
  SELECT
    `bfcol_0`,
    `bfcol_1`,
    ANY_VALUE(`bfcol_49`) AS `bfcol_53`
  FROM `bfcte_15`
  WHERE
    NOT `bfcol_0` IS NULL AND NOT `bfcol_1` IS NULL
  GROUP BY
    `bfcol_0`,
    `bfcol_1`
)
SELECT
  ROUND(IEEE_DIVIDE(`bfcol_53`, 7.0), 2) AS `AVG_YEARLY`
FROM `bfcte_16`
ORDER BY
  `bfcol_1` ASC NULLS LAST,
  `bfcol_0` ASC NULLS LAST