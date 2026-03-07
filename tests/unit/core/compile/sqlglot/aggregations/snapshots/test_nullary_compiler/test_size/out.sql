SELECT
  `bfcol_32` AS `size`
FROM (
  SELECT
    COUNT(1) AS `bfcol_32`
  FROM (
    SELECT
      *
    FROM `bigframes-dev`.`sqlglot_test`.`scalar_types` AS `bft_0`
  )
)