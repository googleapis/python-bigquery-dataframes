WITH `bfcte_2` AS (
  SELECT
    `P_PARTKEY` AS `bfcol_0`,
    `P_TYPE` AS `bfcol_1`,
    `P_SIZE` AS `bfcol_2`
  FROM `bigframes-dev`.`tpch`.`PART` AS `bft_4` FOR SYSTEM_TIME AS OF '2026-03-10T18:00:00'
), `bfcte_3` AS (
  SELECT
    `PS_PARTKEY` AS `bfcol_3`,
    `PS_SUPPKEY` AS `bfcol_4`,
    `PS_SUPPLYCOST` AS `bfcol_5`
  FROM `bigframes-dev`.`tpch`.`PARTSUPP` AS `bft_3` FOR SYSTEM_TIME AS OF '2026-03-10T18:00:00'
), `bfcte_5` AS (
  SELECT
    *
  FROM `bfcte_2`
  INNER JOIN `bfcte_3`
    ON COALESCE(`bfcol_0`, 0) = COALESCE(`bfcol_3`, 0)
    AND COALESCE(`bfcol_0`, 1) = COALESCE(`bfcol_3`, 1)
), `bfcte_8` AS (
  SELECT
    `bfcol_0` AS `bfcol_6`,
    `bfcol_1` AS `bfcol_7`,
    `bfcol_2` AS `bfcol_8`,
    `bfcol_4` AS `bfcol_9`,
    `bfcol_5` AS `bfcol_10`
  FROM `bfcte_5`
), `bfcte_9` AS (
  SELECT
    `S_SUPPKEY` AS `bfcol_11`,
    `S_NATIONKEY` AS `bfcol_12`
  FROM `bigframes-dev`.`tpch`.`SUPPLIER` AS `bft_2` FOR SYSTEM_TIME AS OF '2026-03-10T18:00:00'
), `bfcte_11` AS (
  SELECT
    *
  FROM `bfcte_8`
  INNER JOIN `bfcte_9`
    ON COALESCE(`bfcol_9`, 0) = COALESCE(`bfcol_11`, 0)
    AND COALESCE(`bfcol_9`, 1) = COALESCE(`bfcol_11`, 1)
), `bfcte_14` AS (
  SELECT
    `bfcol_6` AS `bfcol_13`,
    `bfcol_7` AS `bfcol_14`,
    `bfcol_8` AS `bfcol_15`,
    `bfcol_10` AS `bfcol_16`,
    `bfcol_12` AS `bfcol_17`
  FROM `bfcte_11`
), `bfcte_15` AS (
  SELECT
    `N_NATIONKEY` AS `bfcol_18`,
    `N_REGIONKEY` AS `bfcol_19`
  FROM `bigframes-dev`.`tpch`.`NATION` AS `bft_1` FOR SYSTEM_TIME AS OF '2026-03-10T18:00:00'
), `bfcte_17` AS (
  SELECT
    *
  FROM `bfcte_14`
  INNER JOIN `bfcte_15`
    ON COALESCE(`bfcol_17`, 0) = COALESCE(`bfcol_18`, 0)
    AND COALESCE(`bfcol_17`, 1) = COALESCE(`bfcol_18`, 1)
), `bfcte_20` AS (
  SELECT
    `bfcol_13` AS `bfcol_20`,
    `bfcol_14` AS `bfcol_21`,
    `bfcol_15` AS `bfcol_22`,
    `bfcol_16` AS `bfcol_23`,
    `bfcol_19` AS `bfcol_24`
  FROM `bfcte_17`
), `bfcte_21` AS (
  SELECT
    `R_REGIONKEY` AS `bfcol_25`,
    `R_NAME` AS `bfcol_26`
  FROM `bigframes-dev`.`tpch`.`REGION` AS `bft_0` FOR SYSTEM_TIME AS OF '2026-03-10T18:00:00'
), `bfcte_23` AS (
  SELECT
    *
  FROM `bfcte_20`
  INNER JOIN `bfcte_21`
    ON COALESCE(`bfcol_24`, 0) = COALESCE(`bfcol_25`, 0)
    AND COALESCE(`bfcol_24`, 1) = COALESCE(`bfcol_25`, 1)
), `bfcte_24` AS (
  SELECT
    `bfcol_20`,
    `bfcol_21`,
    `bfcol_22`,
    `bfcol_23`,
    `bfcol_24`,
    `bfcol_25`,
    `bfcol_26`,
    `bfcol_20` AS `bfcol_32`,
    `bfcol_21` AS `bfcol_33`,
    `bfcol_23` AS `bfcol_34`,
    `bfcol_26` AS `bfcol_35`,
    `bfcol_22` = 15 AS `bfcol_36`,
    `bfcol_20` AS `bfcol_46`,
    `bfcol_23` AS `bfcol_47`,
    `bfcol_26` AS `bfcol_48`,
    ENDS_WITH(`bfcol_21`, 'BRASS') AS `bfcol_49`,
    `bfcol_20` AS `bfcol_57`,
    `bfcol_23` AS `bfcol_58`,
    `bfcol_26` = 'EUROPE' AS `bfcol_59`
  FROM `bfcte_23`
  WHERE
    `bfcol_22` = 15 AND ENDS_WITH(`bfcol_21`, 'BRASS') AND `bfcol_26` = 'EUROPE'
), `bfcte_25` AS (
  SELECT
    `bfcol_57`,
    MIN(`bfcol_58`) AS `bfcol_65`
  FROM `bfcte_24`
  WHERE
    NOT `bfcol_57` IS NULL
  GROUP BY
    `bfcol_57`
), `bfcte_26` AS (
  SELECT
    `bfcol_57` AS `bfcol_63`,
    `bfcol_65`
  FROM `bfcte_25`
), `bfcte_0` AS (
  SELECT
    `P_PARTKEY` AS `bfcol_66`,
    `P_MFGR` AS `bfcol_67`,
    `P_TYPE` AS `bfcol_68`,
    `P_SIZE` AS `bfcol_69`
  FROM `bigframes-dev`.`tpch`.`PART` AS `bft_4` FOR SYSTEM_TIME AS OF '2026-03-10T18:00:00'
), `bfcte_1` AS (
  SELECT
    `PS_PARTKEY` AS `bfcol_70`,
    `PS_SUPPKEY` AS `bfcol_71`,
    `PS_SUPPLYCOST` AS `bfcol_72`
  FROM `bigframes-dev`.`tpch`.`PARTSUPP` AS `bft_3` FOR SYSTEM_TIME AS OF '2026-03-10T18:00:00'
), `bfcte_4` AS (
  SELECT
    *
  FROM `bfcte_0`
  INNER JOIN `bfcte_1`
    ON COALESCE(`bfcol_66`, 0) = COALESCE(`bfcol_70`, 0)
    AND COALESCE(`bfcol_66`, 1) = COALESCE(`bfcol_70`, 1)
), `bfcte_6` AS (
  SELECT
    `bfcol_66` AS `bfcol_73`,
    `bfcol_67` AS `bfcol_74`,
    `bfcol_68` AS `bfcol_75`,
    `bfcol_69` AS `bfcol_76`,
    `bfcol_71` AS `bfcol_77`,
    `bfcol_72` AS `bfcol_78`
  FROM `bfcte_4`
), `bfcte_7` AS (
  SELECT
    `S_SUPPKEY` AS `bfcol_79`,
    `S_NAME` AS `bfcol_80`,
    `S_ADDRESS` AS `bfcol_81`,
    `S_NATIONKEY` AS `bfcol_82`,
    `S_PHONE` AS `bfcol_83`,
    `S_ACCTBAL` AS `bfcol_84`,
    `S_COMMENT` AS `bfcol_85`
  FROM `bigframes-dev`.`tpch`.`SUPPLIER` AS `bft_2` FOR SYSTEM_TIME AS OF '2026-03-10T18:00:00'
), `bfcte_10` AS (
  SELECT
    *
  FROM `bfcte_6`
  INNER JOIN `bfcte_7`
    ON COALESCE(`bfcol_77`, 0) = COALESCE(`bfcol_79`, 0)
    AND COALESCE(`bfcol_77`, 1) = COALESCE(`bfcol_79`, 1)
), `bfcte_12` AS (
  SELECT
    `bfcol_73` AS `bfcol_86`,
    `bfcol_74` AS `bfcol_87`,
    `bfcol_75` AS `bfcol_88`,
    `bfcol_76` AS `bfcol_89`,
    `bfcol_78` AS `bfcol_90`,
    `bfcol_80` AS `bfcol_91`,
    `bfcol_81` AS `bfcol_92`,
    `bfcol_82` AS `bfcol_93`,
    `bfcol_83` AS `bfcol_94`,
    `bfcol_84` AS `bfcol_95`,
    `bfcol_85` AS `bfcol_96`
  FROM `bfcte_10`
), `bfcte_13` AS (
  SELECT
    `N_NATIONKEY` AS `bfcol_97`,
    `N_NAME` AS `bfcol_98`,
    `N_REGIONKEY` AS `bfcol_99`
  FROM `bigframes-dev`.`tpch`.`NATION` AS `bft_1` FOR SYSTEM_TIME AS OF '2026-03-10T18:00:00'
), `bfcte_16` AS (
  SELECT
    *
  FROM `bfcte_12`
  INNER JOIN `bfcte_13`
    ON COALESCE(`bfcol_93`, 0) = COALESCE(`bfcol_97`, 0)
    AND COALESCE(`bfcol_93`, 1) = COALESCE(`bfcol_97`, 1)
), `bfcte_18` AS (
  SELECT
    `bfcol_86` AS `bfcol_100`,
    `bfcol_87` AS `bfcol_101`,
    `bfcol_88` AS `bfcol_102`,
    `bfcol_89` AS `bfcol_103`,
    `bfcol_90` AS `bfcol_104`,
    `bfcol_91` AS `bfcol_105`,
    `bfcol_92` AS `bfcol_106`,
    `bfcol_94` AS `bfcol_107`,
    `bfcol_95` AS `bfcol_108`,
    `bfcol_96` AS `bfcol_109`,
    `bfcol_98` AS `bfcol_110`,
    `bfcol_99` AS `bfcol_111`
  FROM `bfcte_16`
), `bfcte_19` AS (
  SELECT
    `R_REGIONKEY` AS `bfcol_112`,
    `R_NAME` AS `bfcol_113`
  FROM `bigframes-dev`.`tpch`.`REGION` AS `bft_0` FOR SYSTEM_TIME AS OF '2026-03-10T18:00:00'
), `bfcte_22` AS (
  SELECT
    *
  FROM `bfcte_18`
  INNER JOIN `bfcte_19`
    ON COALESCE(`bfcol_111`, 0) = COALESCE(`bfcol_112`, 0)
    AND COALESCE(`bfcol_111`, 1) = COALESCE(`bfcol_112`, 1)
), `bfcte_27` AS (
  SELECT
    `bfcol_100` AS `bfcol_213`,
    `bfcol_101` AS `bfcol_214`,
    `bfcol_104` AS `bfcol_215`,
    `bfcol_105` AS `bfcol_216`,
    `bfcol_106` AS `bfcol_217`,
    `bfcol_107` AS `bfcol_218`,
    `bfcol_108` AS `bfcol_219`,
    `bfcol_109` AS `bfcol_220`,
    `bfcol_110` AS `bfcol_221`
  FROM `bfcte_22`
  WHERE
    `bfcol_103` = 15 AND ENDS_WITH(`bfcol_102`, 'BRASS') AND `bfcol_113` = 'EUROPE'
), `bfcte_28` AS (
  SELECT
    *
  FROM `bfcte_26`
  INNER JOIN `bfcte_27`
    ON COALESCE(`bfcol_63`, 0) = COALESCE(`bfcol_213`, 0)
    AND COALESCE(`bfcol_63`, 1) = COALESCE(`bfcol_213`, 1)
    AND IF(IS_NAN(`bfcol_65`), 2, COALESCE(`bfcol_65`, 0)) = IF(IS_NAN(`bfcol_215`), 2, COALESCE(`bfcol_215`, 0))
    AND IF(IS_NAN(`bfcol_65`), 3, COALESCE(`bfcol_65`, 1)) = IF(IS_NAN(`bfcol_215`), 3, COALESCE(`bfcol_215`, 1))
)
SELECT
  `bfcol_219` AS `S_ACCTBAL`,
  `bfcol_216` AS `S_NAME`,
  `bfcol_221` AS `N_NAME`,
  `bfcol_63` AS `P_PARTKEY`,
  `bfcol_214` AS `P_MFGR`,
  `bfcol_217` AS `S_ADDRESS`,
  `bfcol_218` AS `S_PHONE`,
  `bfcol_220` AS `S_COMMENT`
FROM `bfcte_28`
ORDER BY
  `bfcol_219` DESC,
  `bfcol_221` ASC NULLS LAST,
  `bfcol_216` ASC NULLS LAST,
  `bfcol_63` ASC NULLS LAST
LIMIT 100