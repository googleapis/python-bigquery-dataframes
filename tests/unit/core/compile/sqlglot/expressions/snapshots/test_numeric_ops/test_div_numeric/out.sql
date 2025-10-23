WITH `bfcte_0` AS (
  SELECT
    *
  FROM UNNEST(ARRAY<STRUCT<`bfcol_0` BOOLEAN, `bfcol_1` INT64, `bfcol_2` FLOAT64, `bfcol_3` INT64, `bfcol_4` INT64>>[STRUCT(
    CAST(NULL AS BOOLEAN),
    CAST(NULL AS INT64),
    CAST(NULL AS FLOAT64),
    CAST(NULL AS INT64),
    0
  )])
), `bfcte_1` AS (
  SELECT
    *,
    `bfcol_3` AS `bfcol_10`,
    `bfcol_1` AS `bfcol_11`,
    `bfcol_0` AS `bfcol_12`,
    `bfcol_2` AS `bfcol_13`,
    IEEE_DIVIDE(`bfcol_1`, `bfcol_1`) AS `bfcol_14`
  FROM `bfcte_0`
), `bfcte_2` AS (
  SELECT
    *,
    `bfcol_10` AS `bfcol_21`,
    `bfcol_11` AS `bfcol_22`,
    `bfcol_12` AS `bfcol_23`,
    `bfcol_13` AS `bfcol_24`,
    `bfcol_14` AS `bfcol_25`,
    IEEE_DIVIDE(`bfcol_11`, 1) AS `bfcol_26`
  FROM `bfcte_1`
), `bfcte_3` AS (
  SELECT
    *,
    `bfcol_21` AS `bfcol_34`,
    `bfcol_22` AS `bfcol_35`,
    `bfcol_23` AS `bfcol_36`,
    `bfcol_24` AS `bfcol_37`,
    `bfcol_25` AS `bfcol_38`,
    `bfcol_26` AS `bfcol_39`,
    IEEE_DIVIDE(`bfcol_22`, 0.0) AS `bfcol_40`
  FROM `bfcte_2`
), `bfcte_4` AS (
  SELECT
    *,
    `bfcol_34` AS `bfcol_49`,
    `bfcol_35` AS `bfcol_50`,
    `bfcol_36` AS `bfcol_51`,
    `bfcol_37` AS `bfcol_52`,
    `bfcol_38` AS `bfcol_53`,
    `bfcol_39` AS `bfcol_54`,
    `bfcol_40` AS `bfcol_55`,
    IEEE_DIVIDE(`bfcol_35`, `bfcol_37`) AS `bfcol_56`
  FROM `bfcte_3`
), `bfcte_5` AS (
  SELECT
    *,
    `bfcol_49` AS `bfcol_66`,
    `bfcol_50` AS `bfcol_67`,
    `bfcol_51` AS `bfcol_68`,
    `bfcol_52` AS `bfcol_69`,
    `bfcol_53` AS `bfcol_70`,
    `bfcol_54` AS `bfcol_71`,
    `bfcol_55` AS `bfcol_72`,
    `bfcol_56` AS `bfcol_73`,
    IEEE_DIVIDE(`bfcol_52`, `bfcol_50`) AS `bfcol_74`
  FROM `bfcte_4`
), `bfcte_6` AS (
  SELECT
    *,
    `bfcol_66` AS `bfcol_85`,
    `bfcol_67` AS `bfcol_86`,
    `bfcol_68` AS `bfcol_87`,
    `bfcol_69` AS `bfcol_88`,
    `bfcol_70` AS `bfcol_89`,
    `bfcol_71` AS `bfcol_90`,
    `bfcol_72` AS `bfcol_91`,
    `bfcol_73` AS `bfcol_92`,
    `bfcol_74` AS `bfcol_93`,
    IEEE_DIVIDE(`bfcol_69`, 0.0) AS `bfcol_94`
  FROM `bfcte_5`
), `bfcte_7` AS (
  SELECT
    *,
    `bfcol_85` AS `bfcol_106`,
    `bfcol_86` AS `bfcol_107`,
    `bfcol_87` AS `bfcol_108`,
    `bfcol_88` AS `bfcol_109`,
    `bfcol_89` AS `bfcol_110`,
    `bfcol_90` AS `bfcol_111`,
    `bfcol_91` AS `bfcol_112`,
    `bfcol_92` AS `bfcol_113`,
    `bfcol_93` AS `bfcol_114`,
    `bfcol_94` AS `bfcol_115`,
    IEEE_DIVIDE(`bfcol_86`, CAST(`bfcol_87` AS INT64)) AS `bfcol_116`
  FROM `bfcte_6`
), `bfcte_8` AS (
  SELECT
    *,
    `bfcol_106` AS `bfcol_129`,
    `bfcol_107` AS `bfcol_130`,
    `bfcol_108` AS `bfcol_131`,
    `bfcol_109` AS `bfcol_132`,
    `bfcol_110` AS `bfcol_133`,
    `bfcol_111` AS `bfcol_134`,
    `bfcol_112` AS `bfcol_135`,
    `bfcol_113` AS `bfcol_136`,
    `bfcol_114` AS `bfcol_137`,
    `bfcol_115` AS `bfcol_138`,
    `bfcol_116` AS `bfcol_139`,
    IEEE_DIVIDE(CAST(`bfcol_108` AS INT64), `bfcol_107`) AS `bfcol_140`
  FROM `bfcte_7`
)
SELECT
  `bfcol_129` AS `rowindex`,
  `bfcol_130` AS `int64_col`,
  `bfcol_131` AS `bool_col`,
  `bfcol_132` AS `float64_col`,
  `bfcol_133` AS `int_div_int`,
  `bfcol_134` AS `int_div_1`,
  `bfcol_135` AS `int_div_0`,
  `bfcol_136` AS `int_div_float`,
  `bfcol_137` AS `float_div_int`,
  `bfcol_138` AS `float_div_0`,
  `bfcol_139` AS `int_div_bool`,
  `bfcol_140` AS `bool_div_int`
FROM `bfcte_8`
ORDER BY
  `bfcol_4` ASC NULLS LAST