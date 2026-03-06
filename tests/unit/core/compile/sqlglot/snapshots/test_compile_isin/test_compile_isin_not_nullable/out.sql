(
  SELECT
    `bfcol_2` AS `rowindex`,
    `bfcol_5` AS `rowindex_2`
  FROM (
    SELECT
      `rowindex` AS `bfcol_2`,
      `rowindex_2` AS `bfcol_3`
    FROM `bigframes-dev`.`sqlglot_test`.`scalar_types` AS `bft_0`
  )
)