WITH `bfcte_12` AS (
  SELECT
    *
  FROM UNNEST(ARRAY<STRUCT<`bfcol_0` STRING, `bfcol_1` INT64, `bfcol_2` INT64>>[STRUCT('TEMP', 0, 0)])
), `bfcte_2` AS (
  SELECT
    `P_PARTKEY` AS `bfcol_3`,
    `P_TYPE` AS `bfcol_4`
  FROM `bigframes-dev`.`tpch`.`PART` AS `bft_1` FOR SYSTEM_TIME AS OF '2026-03-10T18:00:00'
), `bfcte_3` AS (
  SELECT
    `L_PARTKEY` AS `bfcol_5`,
    `L_EXTENDEDPRICE` AS `bfcol_6`,
    `L_DISCOUNT` AS `bfcol_7`,
    `L_SHIPDATE` AS `bfcol_8`
  FROM `bigframes-dev`.`tpch`.`LINEITEM` AS `bft_0` FOR SYSTEM_TIME AS OF '2026-03-10T18:00:00'
), `bfcte_5` AS (
  SELECT
    *
  FROM `bfcte_2`
  INNER JOIN `bfcte_3`
    ON COALESCE(`bfcol_3`, 0) = COALESCE(`bfcol_5`, 0)
    AND COALESCE(`bfcol_3`, 1) = COALESCE(`bfcol_5`, 1)
), `bfcte_7` AS (
  SELECT
    `bfcol_3`,
    `bfcol_4`,
    `bfcol_5`,
    `bfcol_6`,
    `bfcol_7`,
    `bfcol_8`,
    `bfcol_4` AS `bfcol_13`,
    `bfcol_6` AS `bfcol_14`,
    `bfcol_7` AS `bfcol_15`,
    (
      `bfcol_8` >= CAST('1995-09-01' AS DATE)
    )
    AND (
      `bfcol_8` < CAST('1995-10-01' AS DATE)
    ) AS `bfcol_16`,
    (
      `bfcol_6` * (
        1 - `bfcol_7`
      )
    ) * CAST(REGEXP_CONTAINS(`bfcol_4`, 'PROMO') AS INT64) AS `bfcol_24`
  FROM `bfcte_5`
  WHERE
    (
      `bfcol_8` >= CAST('1995-09-01' AS DATE)
    )
    AND (
      `bfcol_8` < CAST('1995-10-01' AS DATE)
    )
), `bfcte_9` AS (
  SELECT
    COALESCE(SUM(`bfcol_24`), 0) AS `bfcol_26`
  FROM `bfcte_7`
), `bfcte_13` AS (
  SELECT
    `bfcol_26`,
    0 AS `bfcol_27`
  FROM `bfcte_9`
), `bfcte_15` AS (
  SELECT
    *
  FROM `bfcte_12`
  CROSS JOIN `bfcte_13`
), `bfcte_17` AS (
  SELECT
    `bfcol_0`,
    `bfcol_1`,
    `bfcol_2`,
    `bfcol_26`,
    `bfcol_27`,
    CASE WHEN `bfcol_2` = 0 THEN `bfcol_26` END AS `bfcol_28`,
    IF(`bfcol_27` = 0, CASE WHEN `bfcol_2` = 0 THEN `bfcol_26` END, NULL) AS `bfcol_33`
  FROM `bfcte_15`
), `bfcte_19` AS (
  SELECT
    `bfcol_0`,
    `bfcol_1`,
    ANY_VALUE(`bfcol_33`) AS `bfcol_37`
  FROM `bfcte_17`
  WHERE
    NOT `bfcol_0` IS NULL AND NOT `bfcol_1` IS NULL
  GROUP BY
    `bfcol_0`,
    `bfcol_1`
), `bfcte_20` AS (
  SELECT
    `bfcol_0` AS `bfcol_41`,
    100.0 * `bfcol_37` AS `bfcol_42`
  FROM `bfcte_19`
), `bfcte_10` AS (
  SELECT
    *
  FROM UNNEST(ARRAY<STRUCT<`bfcol_43` STRING, `bfcol_44` INT64, `bfcol_45` INT64>>[STRUCT('TEMP', 0, 0)])
), `bfcte_0` AS (
  SELECT
    `P_PARTKEY` AS `bfcol_46`
  FROM `bigframes-dev`.`tpch`.`PART` AS `bft_1` FOR SYSTEM_TIME AS OF '2026-03-10T18:00:00'
), `bfcte_1` AS (
  SELECT
    `L_PARTKEY` AS `bfcol_47`,
    `L_EXTENDEDPRICE` AS `bfcol_48`,
    `L_DISCOUNT` AS `bfcol_49`,
    `L_SHIPDATE` AS `bfcol_50`
  FROM `bigframes-dev`.`tpch`.`LINEITEM` AS `bft_0` FOR SYSTEM_TIME AS OF '2026-03-10T18:00:00'
), `bfcte_4` AS (
  SELECT
    *
  FROM `bfcte_0`
  INNER JOIN `bfcte_1`
    ON COALESCE(`bfcol_46`, 0) = COALESCE(`bfcol_47`, 0)
    AND COALESCE(`bfcol_46`, 1) = COALESCE(`bfcol_47`, 1)
), `bfcte_6` AS (
  SELECT
    `bfcol_46`,
    `bfcol_47`,
    `bfcol_48`,
    `bfcol_49`,
    `bfcol_50`,
    `bfcol_48` AS `bfcol_54`,
    `bfcol_49` AS `bfcol_55`,
    (
      `bfcol_50` >= CAST('1995-09-01' AS DATE)
    )
    AND (
      `bfcol_50` < CAST('1995-10-01' AS DATE)
    ) AS `bfcol_56`,
    `bfcol_48` AS `bfcol_62`,
    `bfcol_49` AS `bfcol_63`,
    `bfcol_48` AS `bfcol_66`,
    1 - `bfcol_49` AS `bfcol_67`,
    `bfcol_48` * (
      1 - `bfcol_49`
    ) AS `bfcol_70`
  FROM `bfcte_4`
  WHERE
    (
      `bfcol_50` >= CAST('1995-09-01' AS DATE)
    )
    AND (
      `bfcol_50` < CAST('1995-10-01' AS DATE)
    )
), `bfcte_8` AS (
  SELECT
    COALESCE(SUM(`bfcol_70`), 0) AS `bfcol_72`
  FROM `bfcte_6`
), `bfcte_11` AS (
  SELECT
    `bfcol_72`,
    0 AS `bfcol_73`
  FROM `bfcte_8`
), `bfcte_14` AS (
  SELECT
    *
  FROM `bfcte_10`
  CROSS JOIN `bfcte_11`
), `bfcte_16` AS (
  SELECT
    `bfcol_43`,
    `bfcol_44`,
    `bfcol_45`,
    `bfcol_72`,
    `bfcol_73`,
    CASE WHEN `bfcol_45` = 0 THEN `bfcol_72` END AS `bfcol_74`,
    IF(`bfcol_73` = 0, CASE WHEN `bfcol_45` = 0 THEN `bfcol_72` END, NULL) AS `bfcol_79`
  FROM `bfcte_14`
), `bfcte_18` AS (
  SELECT
    `bfcol_43`,
    `bfcol_44`,
    ANY_VALUE(`bfcol_79`) AS `bfcol_83`
  FROM `bfcte_16`
  WHERE
    NOT `bfcol_43` IS NULL AND NOT `bfcol_44` IS NULL
  GROUP BY
    `bfcol_43`,
    `bfcol_44`
), `bfcte_21` AS (
  SELECT
    `bfcol_43` AS `bfcol_84`,
    `bfcol_83` AS `bfcol_85`
  FROM `bfcte_18`
), `bfcte_22` AS (
  SELECT
    *
  FROM `bfcte_20`
  FULL OUTER JOIN `bfcte_21`
    ON `bfcol_41` = `bfcol_84`
)
SELECT
  ROUND(IEEE_DIVIDE(`bfcol_42`, `bfcol_85`), 2) AS `PROMO_REVENUE`
FROM `bfcte_22`
ORDER BY
  COALESCE(`bfcol_41`, `bfcol_84`) ASC NULLS LAST