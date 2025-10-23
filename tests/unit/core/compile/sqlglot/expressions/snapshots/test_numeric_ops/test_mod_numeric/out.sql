WITH `bfcte_0` AS (
  SELECT
    *
  FROM UNNEST(ARRAY<STRUCT<`bfcol_0` INT64, `bfcol_1` FLOAT64, `bfcol_2` INT64, `bfcol_3` INT64>>[STRUCT(CAST(NULL AS INT64), CAST(NULL AS FLOAT64), CAST(NULL AS INT64), 0)])
), `bfcte_1` AS (
  SELECT
    *,
    `bfcol_2` AS `bfcol_8`,
    `bfcol_0` AS `bfcol_9`,
    `bfcol_1` AS `bfcol_10`,
    CASE
      WHEN `bfcol_0` = CAST(0 AS INT64)
      THEN CAST(0 AS INT64) * `bfcol_0`
      WHEN `bfcol_0` < CAST(0 AS INT64)
      AND (
        MOD(`bfcol_0`, `bfcol_0`)
      ) > CAST(0 AS INT64)
      THEN `bfcol_0` + (
        MOD(`bfcol_0`, `bfcol_0`)
      )
      WHEN `bfcol_0` > CAST(0 AS INT64)
      AND (
        MOD(`bfcol_0`, `bfcol_0`)
      ) < CAST(0 AS INT64)
      THEN `bfcol_0` + (
        MOD(`bfcol_0`, `bfcol_0`)
      )
      ELSE MOD(`bfcol_0`, `bfcol_0`)
    END AS `bfcol_11`
  FROM `bfcte_0`
), `bfcte_2` AS (
  SELECT
    *,
    `bfcol_8` AS `bfcol_17`,
    `bfcol_9` AS `bfcol_18`,
    `bfcol_10` AS `bfcol_19`,
    `bfcol_11` AS `bfcol_20`,
    CASE
      WHEN -`bfcol_9` = CAST(0 AS INT64)
      THEN CAST(0 AS INT64) * `bfcol_9`
      WHEN -`bfcol_9` < CAST(0 AS INT64)
      AND (
        MOD(`bfcol_9`, -`bfcol_9`)
      ) > CAST(0 AS INT64)
      THEN -`bfcol_9` + (
        MOD(`bfcol_9`, -`bfcol_9`)
      )
      WHEN -`bfcol_9` > CAST(0 AS INT64)
      AND (
        MOD(`bfcol_9`, -`bfcol_9`)
      ) < CAST(0 AS INT64)
      THEN -`bfcol_9` + (
        MOD(`bfcol_9`, -`bfcol_9`)
      )
      ELSE MOD(`bfcol_9`, -`bfcol_9`)
    END AS `bfcol_21`
  FROM `bfcte_1`
), `bfcte_3` AS (
  SELECT
    *,
    `bfcol_17` AS `bfcol_28`,
    `bfcol_18` AS `bfcol_29`,
    `bfcol_19` AS `bfcol_30`,
    `bfcol_20` AS `bfcol_31`,
    `bfcol_21` AS `bfcol_32`,
    CASE
      WHEN 1 = CAST(0 AS INT64)
      THEN CAST(0 AS INT64) * `bfcol_18`
      WHEN 1 < CAST(0 AS INT64) AND (
        MOD(`bfcol_18`, 1)
      ) > CAST(0 AS INT64)
      THEN 1 + (
        MOD(`bfcol_18`, 1)
      )
      WHEN 1 > CAST(0 AS INT64) AND (
        MOD(`bfcol_18`, 1)
      ) < CAST(0 AS INT64)
      THEN 1 + (
        MOD(`bfcol_18`, 1)
      )
      ELSE MOD(`bfcol_18`, 1)
    END AS `bfcol_33`
  FROM `bfcte_2`
), `bfcte_4` AS (
  SELECT
    *,
    `bfcol_28` AS `bfcol_41`,
    `bfcol_29` AS `bfcol_42`,
    `bfcol_30` AS `bfcol_43`,
    `bfcol_31` AS `bfcol_44`,
    `bfcol_32` AS `bfcol_45`,
    `bfcol_33` AS `bfcol_46`,
    CASE
      WHEN 0 = CAST(0 AS INT64)
      THEN CAST(0 AS INT64) * `bfcol_29`
      WHEN 0 < CAST(0 AS INT64) AND (
        MOD(`bfcol_29`, 0)
      ) > CAST(0 AS INT64)
      THEN 0 + (
        MOD(`bfcol_29`, 0)
      )
      WHEN 0 > CAST(0 AS INT64) AND (
        MOD(`bfcol_29`, 0)
      ) < CAST(0 AS INT64)
      THEN 0 + (
        MOD(`bfcol_29`, 0)
      )
      ELSE MOD(`bfcol_29`, 0)
    END AS `bfcol_47`
  FROM `bfcte_3`
), `bfcte_5` AS (
  SELECT
    *,
    `bfcol_41` AS `bfcol_56`,
    `bfcol_42` AS `bfcol_57`,
    `bfcol_43` AS `bfcol_58`,
    `bfcol_44` AS `bfcol_59`,
    `bfcol_45` AS `bfcol_60`,
    `bfcol_46` AS `bfcol_61`,
    `bfcol_47` AS `bfcol_62`,
    CASE
      WHEN CAST(`bfcol_43` AS BIGNUMERIC) = CAST(0 AS INT64)
      THEN CAST('NaN' AS FLOAT64) * CAST(`bfcol_43` AS BIGNUMERIC)
      WHEN CAST(`bfcol_43` AS BIGNUMERIC) < CAST(0 AS INT64)
      AND (
        MOD(CAST(`bfcol_43` AS BIGNUMERIC), CAST(`bfcol_43` AS BIGNUMERIC))
      ) > CAST(0 AS INT64)
      THEN CAST(`bfcol_43` AS BIGNUMERIC) + (
        MOD(CAST(`bfcol_43` AS BIGNUMERIC), CAST(`bfcol_43` AS BIGNUMERIC))
      )
      WHEN CAST(`bfcol_43` AS BIGNUMERIC) > CAST(0 AS INT64)
      AND (
        MOD(CAST(`bfcol_43` AS BIGNUMERIC), CAST(`bfcol_43` AS BIGNUMERIC))
      ) < CAST(0 AS INT64)
      THEN CAST(`bfcol_43` AS BIGNUMERIC) + (
        MOD(CAST(`bfcol_43` AS BIGNUMERIC), CAST(`bfcol_43` AS BIGNUMERIC))
      )
      ELSE MOD(CAST(`bfcol_43` AS BIGNUMERIC), CAST(`bfcol_43` AS BIGNUMERIC))
    END AS `bfcol_63`
  FROM `bfcte_4`
), `bfcte_6` AS (
  SELECT
    *,
    `bfcol_56` AS `bfcol_73`,
    `bfcol_57` AS `bfcol_74`,
    `bfcol_58` AS `bfcol_75`,
    `bfcol_59` AS `bfcol_76`,
    `bfcol_60` AS `bfcol_77`,
    `bfcol_61` AS `bfcol_78`,
    `bfcol_62` AS `bfcol_79`,
    `bfcol_63` AS `bfcol_80`,
    CASE
      WHEN CAST(-`bfcol_58` AS BIGNUMERIC) = CAST(0 AS INT64)
      THEN CAST('NaN' AS FLOAT64) * CAST(`bfcol_58` AS BIGNUMERIC)
      WHEN CAST(-`bfcol_58` AS BIGNUMERIC) < CAST(0 AS INT64)
      AND (
        MOD(CAST(`bfcol_58` AS BIGNUMERIC), CAST(-`bfcol_58` AS BIGNUMERIC))
      ) > CAST(0 AS INT64)
      THEN CAST(-`bfcol_58` AS BIGNUMERIC) + (
        MOD(CAST(`bfcol_58` AS BIGNUMERIC), CAST(-`bfcol_58` AS BIGNUMERIC))
      )
      WHEN CAST(-`bfcol_58` AS BIGNUMERIC) > CAST(0 AS INT64)
      AND (
        MOD(CAST(`bfcol_58` AS BIGNUMERIC), CAST(-`bfcol_58` AS BIGNUMERIC))
      ) < CAST(0 AS INT64)
      THEN CAST(-`bfcol_58` AS BIGNUMERIC) + (
        MOD(CAST(`bfcol_58` AS BIGNUMERIC), CAST(-`bfcol_58` AS BIGNUMERIC))
      )
      ELSE MOD(CAST(`bfcol_58` AS BIGNUMERIC), CAST(-`bfcol_58` AS BIGNUMERIC))
    END AS `bfcol_81`
  FROM `bfcte_5`
), `bfcte_7` AS (
  SELECT
    *,
    `bfcol_73` AS `bfcol_92`,
    `bfcol_74` AS `bfcol_93`,
    `bfcol_75` AS `bfcol_94`,
    `bfcol_76` AS `bfcol_95`,
    `bfcol_77` AS `bfcol_96`,
    `bfcol_78` AS `bfcol_97`,
    `bfcol_79` AS `bfcol_98`,
    `bfcol_80` AS `bfcol_99`,
    `bfcol_81` AS `bfcol_100`,
    CASE
      WHEN CAST(1 AS BIGNUMERIC) = CAST(0 AS INT64)
      THEN CAST('NaN' AS FLOAT64) * CAST(`bfcol_75` AS BIGNUMERIC)
      WHEN CAST(1 AS BIGNUMERIC) < CAST(0 AS INT64)
      AND (
        MOD(CAST(`bfcol_75` AS BIGNUMERIC), CAST(1 AS BIGNUMERIC))
      ) > CAST(0 AS INT64)
      THEN CAST(1 AS BIGNUMERIC) + (
        MOD(CAST(`bfcol_75` AS BIGNUMERIC), CAST(1 AS BIGNUMERIC))
      )
      WHEN CAST(1 AS BIGNUMERIC) > CAST(0 AS INT64)
      AND (
        MOD(CAST(`bfcol_75` AS BIGNUMERIC), CAST(1 AS BIGNUMERIC))
      ) < CAST(0 AS INT64)
      THEN CAST(1 AS BIGNUMERIC) + (
        MOD(CAST(`bfcol_75` AS BIGNUMERIC), CAST(1 AS BIGNUMERIC))
      )
      ELSE MOD(CAST(`bfcol_75` AS BIGNUMERIC), CAST(1 AS BIGNUMERIC))
    END AS `bfcol_101`
  FROM `bfcte_6`
), `bfcte_8` AS (
  SELECT
    *,
    `bfcol_92` AS `bfcol_113`,
    `bfcol_93` AS `bfcol_114`,
    `bfcol_94` AS `bfcol_115`,
    `bfcol_95` AS `bfcol_116`,
    `bfcol_96` AS `bfcol_117`,
    `bfcol_97` AS `bfcol_118`,
    `bfcol_98` AS `bfcol_119`,
    `bfcol_99` AS `bfcol_120`,
    `bfcol_100` AS `bfcol_121`,
    `bfcol_101` AS `bfcol_122`,
    CASE
      WHEN CAST(0 AS BIGNUMERIC) = CAST(0 AS INT64)
      THEN CAST('NaN' AS FLOAT64) * CAST(`bfcol_94` AS BIGNUMERIC)
      WHEN CAST(0 AS BIGNUMERIC) < CAST(0 AS INT64)
      AND (
        MOD(CAST(`bfcol_94` AS BIGNUMERIC), CAST(0 AS BIGNUMERIC))
      ) > CAST(0 AS INT64)
      THEN CAST(0 AS BIGNUMERIC) + (
        MOD(CAST(`bfcol_94` AS BIGNUMERIC), CAST(0 AS BIGNUMERIC))
      )
      WHEN CAST(0 AS BIGNUMERIC) > CAST(0 AS INT64)
      AND (
        MOD(CAST(`bfcol_94` AS BIGNUMERIC), CAST(0 AS BIGNUMERIC))
      ) < CAST(0 AS INT64)
      THEN CAST(0 AS BIGNUMERIC) + (
        MOD(CAST(`bfcol_94` AS BIGNUMERIC), CAST(0 AS BIGNUMERIC))
      )
      ELSE MOD(CAST(`bfcol_94` AS BIGNUMERIC), CAST(0 AS BIGNUMERIC))
    END AS `bfcol_123`
  FROM `bfcte_7`
)
SELECT
  `bfcol_113` AS `rowindex`,
  `bfcol_114` AS `int64_col`,
  `bfcol_115` AS `float64_col`,
  `bfcol_116` AS `int_mod_int`,
  `bfcol_117` AS `int_mod_int_neg`,
  `bfcol_118` AS `int_mod_1`,
  `bfcol_119` AS `int_mod_0`,
  `bfcol_120` AS `float_mod_float`,
  `bfcol_121` AS `float_mod_float_neg`,
  `bfcol_122` AS `float_mod_1`,
  `bfcol_123` AS `float_mod_0`
FROM `bfcte_8`
ORDER BY
  `bfcol_3` ASC NULLS LAST