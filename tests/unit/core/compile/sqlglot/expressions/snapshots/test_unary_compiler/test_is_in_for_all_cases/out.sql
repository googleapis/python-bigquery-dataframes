WITH `bfcte_0` AS (
  SELECT
    `bool_col` AS `bfcol_0`,
    `bytes_col` AS `bfcol_1`,
    `date_col` AS `bfcol_2`,
    `datetime_col` AS `bfcol_3`,
    `geography_col` AS `bfcol_4`,
    `int64_col` AS `bfcol_5`,
    `int64_too` AS `bfcol_6`,
    `numeric_col` AS `bfcol_7`,
    `float64_col` AS `bfcol_8`,
    `rowindex` AS `bfcol_9`,
    `rowindex_2` AS `bfcol_10`,
    `string_col` AS `bfcol_11`,
    `time_col` AS `bfcol_12`,
    `timestamp_col` AS `bfcol_13`,
    `duration_col` AS `bfcol_14`
  FROM `bigframes-dev`.`sqlglot_test`.`scalar_types`
), `bfcte_1` AS (
  SELECT
    *,
    COALESCE(`bfcol_5` IN (1, 2, 3), FALSE) AS `bfcol_31`,
    (
      `bfcol_5` IS NULL
    ) OR `bfcol_5` IN (123456) AS `bfcol_32`,
    COALESCE(`bfcol_5` IN (123456), FALSE) AS `bfcol_33`,
    COALESCE(`bfcol_5` IN (1.0, 2.0, 3.0), FALSE) AS `bfcol_34`,
    FALSE AS `bfcol_35`,
    COALESCE(`bfcol_5` IN (2.5, 3), FALSE) AS `bfcol_36`,
    FALSE AS `bfcol_37`,
    (
      `bfcol_8` IS NULL
    ) OR `bfcol_8` IN (1, 2, 3) AS `bfcol_38`
  FROM `bfcte_0`
)
SELECT
  `bfcol_9` AS `bfuid_col_1`,
  `bfcol_0` AS `bool_col`,
  `bfcol_1` AS `bytes_col`,
  `bfcol_2` AS `date_col`,
  `bfcol_3` AS `datetime_col`,
  `bfcol_4` AS `geography_col`,
  `bfcol_5` AS `int64_col`,
  `bfcol_6` AS `int64_too`,
  `bfcol_7` AS `numeric_col`,
  `bfcol_8` AS `float64_col`,
  `bfcol_9` AS `rowindex`,
  `bfcol_10` AS `rowindex_2`,
  `bfcol_11` AS `string_col`,
  `bfcol_12` AS `time_col`,
  `bfcol_13` AS `timestamp_col`,
  `bfcol_14` AS `duration_col`,
  `bfcol_31` AS `int in ints`,
  `bfcol_32` AS `int in ints w null`,
  `bfcol_33` AS `int in ints w null wo match nulls`,
  `bfcol_34` AS `int in floats`,
  `bfcol_35` AS `int in strings`,
  `bfcol_36` AS `int in mixed`,
  `bfcol_37` AS `int in empty`,
  `bfcol_38` AS `float in ints`
FROM `bfcte_1`