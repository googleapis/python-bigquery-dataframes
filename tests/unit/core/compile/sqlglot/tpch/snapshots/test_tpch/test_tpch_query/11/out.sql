WITH `bfcte_2` AS (
  SELECT
    `N_NATIONKEY` AS `bfcol_6`
  FROM `bigframes-dev`.`tpch`.`NATION` AS `bft_2` FOR SYSTEM_TIME AS OF '2026-03-10T18:00:00'
  WHERE
    `N_NAME` = 'GERMANY'
), `bfcte_3` AS (
  SELECT
    `S_SUPPKEY` AS `bfcol_7`,
    `S_NATIONKEY` AS `bfcol_8`
  FROM `bigframes-dev`.`tpch`.`SUPPLIER` AS `bft_1` FOR SYSTEM_TIME AS OF '2026-03-10T18:00:00'
), `bfcte_5` AS (
  SELECT
    *
  FROM `bfcte_2`
  INNER JOIN `bfcte_3`
    ON COALESCE(`bfcol_6`, 0) = COALESCE(`bfcol_8`, 0)
    AND COALESCE(`bfcol_6`, 1) = COALESCE(`bfcol_8`, 1)
), `bfcte_8` AS (
  SELECT
    `bfcol_7` AS `bfcol_9`
  FROM `bfcte_5`
), `bfcte_9` AS (
  SELECT
    `PS_PARTKEY` AS `bfcol_10`,
    `PS_SUPPKEY` AS `bfcol_11`,
    `PS_AVAILQTY` AS `bfcol_12`,
    `PS_SUPPLYCOST` AS `bfcol_13`
  FROM `bigframes-dev`.`tpch`.`PARTSUPP` AS `bft_0` FOR SYSTEM_TIME AS OF '2026-03-10T18:00:00'
), `bfcte_11` AS (
  SELECT
    *
  FROM `bfcte_8`
  INNER JOIN `bfcte_9`
    ON COALESCE(`bfcol_9`, 0) = COALESCE(`bfcol_11`, 0)
    AND COALESCE(`bfcol_9`, 1) = COALESCE(`bfcol_11`, 1)
), `bfcte_13` AS (
  SELECT
    `bfcol_9`,
    `bfcol_10`,
    `bfcol_11`,
    `bfcol_12`,
    `bfcol_13`,
    `bfcol_10` AS `bfcol_17`,
    `bfcol_13` * `bfcol_12` AS `bfcol_18`
  FROM `bfcte_11`
), `bfcte_15` AS (
  SELECT
    `bfcol_17`,
    COALESCE(SUM(`bfcol_18`), 0) AS `bfcol_21`
  FROM `bfcte_13`
  WHERE
    NOT `bfcol_17` IS NULL
  GROUP BY
    `bfcol_17`
), `bfcte_21` AS (
  SELECT
    `bfcol_17` AS `bfcol_24`,
    ROUND(`bfcol_21`, 2) AS `bfcol_25`
  FROM `bfcte_15`
), `bfcte_16` AS (
  SELECT
    *
  FROM UNNEST(ARRAY<STRUCT<`bfcol_26` FLOAT64, `bfcol_27` INT64, `bfcol_28` INT64>>[STRUCT(0.0, 0, 0)])
), `bfcte_0` AS (
  SELECT
    `N_NATIONKEY` AS `bfcol_35`
  FROM `bigframes-dev`.`tpch`.`NATION` AS `bft_2` FOR SYSTEM_TIME AS OF '2026-03-10T18:00:00'
  WHERE
    `N_NAME` = 'GERMANY'
), `bfcte_1` AS (
  SELECT
    `S_SUPPKEY` AS `bfcol_36`,
    `S_NATIONKEY` AS `bfcol_37`
  FROM `bigframes-dev`.`tpch`.`SUPPLIER` AS `bft_1` FOR SYSTEM_TIME AS OF '2026-03-10T18:00:00'
), `bfcte_4` AS (
  SELECT
    *
  FROM `bfcte_0`
  INNER JOIN `bfcte_1`
    ON COALESCE(`bfcol_35`, 0) = COALESCE(`bfcol_37`, 0)
    AND COALESCE(`bfcol_35`, 1) = COALESCE(`bfcol_37`, 1)
), `bfcte_6` AS (
  SELECT
    `bfcol_36` AS `bfcol_38`
  FROM `bfcte_4`
), `bfcte_7` AS (
  SELECT
    `PS_SUPPKEY` AS `bfcol_39`,
    `PS_AVAILQTY` AS `bfcol_40`,
    `PS_SUPPLYCOST` AS `bfcol_41`
  FROM `bigframes-dev`.`tpch`.`PARTSUPP` AS `bft_0` FOR SYSTEM_TIME AS OF '2026-03-10T18:00:00'
), `bfcte_10` AS (
  SELECT
    *
  FROM `bfcte_6`
  INNER JOIN `bfcte_7`
    ON COALESCE(`bfcol_38`, 0) = COALESCE(`bfcol_39`, 0)
    AND COALESCE(`bfcol_38`, 1) = COALESCE(`bfcol_39`, 1)
), `bfcte_12` AS (
  SELECT
    `bfcol_38`,
    `bfcol_39`,
    `bfcol_40`,
    `bfcol_41`,
    `bfcol_40` AS `bfcol_44`,
    `bfcol_41` AS `bfcol_45`,
    `bfcol_41` AS `bfcol_48`,
    `bfcol_40` AS `bfcol_49`,
    `bfcol_41` * `bfcol_40` AS `bfcol_52`
  FROM `bfcte_10`
), `bfcte_14` AS (
  SELECT
    COALESCE(SUM(`bfcol_52`), 0) AS `bfcol_54`
  FROM `bfcte_12`
), `bfcte_17` AS (
  SELECT
    `bfcol_54`,
    0 AS `bfcol_55`
  FROM `bfcte_14`
), `bfcte_18` AS (
  SELECT
    *
  FROM `bfcte_16`
  CROSS JOIN `bfcte_17`
), `bfcte_19` AS (
  SELECT
    `bfcol_26`,
    `bfcol_27`,
    `bfcol_28`,
    `bfcol_54`,
    `bfcol_55`,
    CASE WHEN `bfcol_28` = 0 THEN `bfcol_54` END AS `bfcol_56`,
    IF(`bfcol_55` = 0, CASE WHEN `bfcol_28` = 0 THEN `bfcol_54` END, NULL) AS `bfcol_61`
  FROM `bfcte_18`
), `bfcte_20` AS (
  SELECT
    `bfcol_26`,
    `bfcol_27`,
    ANY_VALUE(`bfcol_61`) AS `bfcol_65`
  FROM `bfcte_19`
  WHERE
    NOT `bfcol_26` IS NULL AND NOT `bfcol_27` IS NULL
  GROUP BY
    `bfcol_26`,
    `bfcol_27`
), `bfcte_22` AS (
  SELECT
    `bfcol_65` * 0.0001 AS `bfcol_68`
  FROM `bfcte_20`
), `bfcte_23` AS (
  SELECT
    *
  FROM `bfcte_21`
  CROSS JOIN `bfcte_22`
)
SELECT
  `bfcol_24` AS `PS_PARTKEY`,
  `bfcol_25` AS `VALUE`
FROM `bfcte_23`
WHERE
  `bfcol_25` > `bfcol_68`
ORDER BY
  `bfcol_25` DESC