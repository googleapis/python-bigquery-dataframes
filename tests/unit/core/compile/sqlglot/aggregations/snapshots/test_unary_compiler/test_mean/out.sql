SELECT
FROM (
  SELECT
    `bfcol_12` AS `int64_col`,
    `bfcol_13` AS `bool_col`,
    `bfcol_14` AS `duration_col`,
    `bfcol_15` AS `int64_col_w_floor`
  FROM `bigframes-dev`.`sqlglot_test`.`scalar_types` AS `bft_0`
)