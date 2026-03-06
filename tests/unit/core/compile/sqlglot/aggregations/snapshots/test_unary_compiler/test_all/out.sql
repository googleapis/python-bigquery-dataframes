SELECT
  (
    SELECT
      `bfcol_2` AS `bool_col`,
      `bfcol_3` AS `int64_col`
    FROM `bigframes-dev`.`sqlglot_test`.`scalar_types` AS `bft_0`
  ).*
FROM (
  SELECT
    `bfcol_2` AS `bool_col`,
    `bfcol_3` AS `int64_col`
  FROM `bigframes-dev`.`sqlglot_test`.`scalar_types` AS `bft_0`
)