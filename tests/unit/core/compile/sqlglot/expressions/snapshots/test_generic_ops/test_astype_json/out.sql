WITH `bfcte_0` AS (
  SELECT
    *
  FROM UNNEST(ARRAY<STRUCT<`bfcol_0` BOOLEAN, `bfcol_1` INT64, `bfcol_2` FLOAT64, `bfcol_3` STRING, `bfcol_4` INT64>>[STRUCT(
    CAST(NULL AS BOOLEAN),
    CAST(NULL AS INT64),
    CAST(NULL AS FLOAT64),
    CAST(NULL AS STRING),
    0
  )])
), `bfcte_1` AS (
  SELECT
    *,
    PARSE_JSON(CAST(`bfcol_1` AS STRING)) AS `bfcol_5`,
    PARSE_JSON(CAST(`bfcol_2` AS STRING)) AS `bfcol_6`,
    PARSE_JSON(CAST(`bfcol_0` AS STRING)) AS `bfcol_7`,
    PARSE_JSON(`bfcol_3`) AS `bfcol_8`,
    PARSE_JSON(CAST(`bfcol_0` AS STRING)) AS `bfcol_9`,
    PARSE_JSON_IN_SAFE(`bfcol_3`) AS `bfcol_10`
  FROM `bfcte_0`
)
SELECT
  `bfcol_5` AS `int64_col`,
  `bfcol_6` AS `float64_col`,
  `bfcol_7` AS `bool_col`,
  `bfcol_8` AS `string_col`,
  `bfcol_9` AS `bool_w_safe`,
  `bfcol_10` AS `string_w_safe`
FROM `bfcte_1`
ORDER BY
  `bfcol_4` ASC NULLS LAST