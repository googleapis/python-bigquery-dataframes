(
  SELECT
    `bfcol_3` AS `bool_col`,
    `bfcol_6` AS `int64_too`
  FROM `bigframes-dev`.`sqlglot_test`.`scalar_types` AS `bft_0`
  GROUP BY
    `bfcol_3`
)
ORDER BY
  `bfcol_3` ASC NULLS LAST