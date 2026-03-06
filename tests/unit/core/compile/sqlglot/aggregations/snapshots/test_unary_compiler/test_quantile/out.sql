SELECT
  (
    SELECT
      `bfcol_4` AS `int64`,
      `bfcol_5` AS `bool`,
      `bfcol_6` AS `int64_w_floor`
    FROM `bigframes-dev`.`sqlglot_test`.`scalar_types` AS `bft_0`
  ).*
FROM (
  SELECT
    `bfcol_4` AS `int64`,
    `bfcol_5` AS `bool`,
    `bfcol_6` AS `int64_w_floor`
  FROM `bigframes-dev`.`sqlglot_test`.`scalar_types` AS `bft_0`
)