SELECT
FROM (
  SELECT
    `bfcol_2` AS `rowindex`,
    `bfcol_5` AS `int64_col`
  FROM (
    SELECT
      `rowindex` AS `bfcol_2`,
      `int64_col` AS `bfcol_3`
    FROM `bigframes-dev`.`sqlglot_test`.`scalar_types` AS `bft_0`
  )
)