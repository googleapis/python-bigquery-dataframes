WITH `bfcte_0` AS (
  SELECT
    `bytes_col` AS `bfcol_0`,
    `date_col` AS `bfcol_1`,
    `datetime_col` AS `bfcol_2`,
    `int64_col` AS `bfcol_3`,
    `string_col` AS `bfcol_4`,
    `time_col` AS `bfcol_5`,
    `timestamp_col` AS `bfcol_6`
  FROM `bigframes-dev`.`sqlglot_test`.`scalar_types`
), `bfcte_1` AS (
  SELECT
    APPROX_QUANTILES(`bfcol_3`, 2)[OFFSET(1)] AS `bfcol_7`,
    APPROX_QUANTILES(`bfcol_0`, 2)[OFFSET(1)] AS `bfcol_8`,
    APPROX_QUANTILES(`bfcol_1`, 2)[OFFSET(1)] AS `bfcol_9`,
    APPROX_QUANTILES(`bfcol_2`, 2)[OFFSET(1)] AS `bfcol_10`,
    APPROX_QUANTILES(`bfcol_5`, 2)[OFFSET(1)] AS `bfcol_11`,
    APPROX_QUANTILES(`bfcol_6`, 2)[OFFSET(1)] AS `bfcol_12`,
    APPROX_QUANTILES(`bfcol_4`, 2)[OFFSET(1)] AS `bfcol_13`
  FROM `bfcte_0`
)
SELECT
  `bfcol_7` AS `int64_col`,
  `bfcol_8` AS `bytes_col`,
  `bfcol_9` AS `date_col`,
  `bfcol_10` AS `datetime_col`,
  `bfcol_11` AS `time_col`,
  `bfcol_12` AS `timestamp_col`,
  `bfcol_13` AS `string_col`
FROM `bfcte_1`