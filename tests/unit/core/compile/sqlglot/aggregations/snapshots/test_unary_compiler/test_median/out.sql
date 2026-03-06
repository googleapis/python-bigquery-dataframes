SELECT
  (
    SELECT
      `bfcol_3` AS `int64_col`,
      `bfcol_4` AS `date_col`,
      `bfcol_5` AS `string_col`
    FROM `bigframes-dev`.`sqlglot_test`.`scalar_types` AS `bft_0`
  ).*
FROM (
  SELECT
    `bfcol_3` AS `int64_col`,
    `bfcol_4` AS `date_col`,
    `bfcol_5` AS `string_col`
  FROM `bigframes-dev`.`sqlglot_test`.`scalar_types` AS `bft_0`
)