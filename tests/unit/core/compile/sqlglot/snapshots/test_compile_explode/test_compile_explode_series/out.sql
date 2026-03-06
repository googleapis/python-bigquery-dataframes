SELECT
FROM (
  SELECT
    `rowindex`,
    `int_list_col`
  FROM `bigframes-dev`.`sqlglot_test`.`repeated_types` AS `bft_0`
  LEFT JOIN UNNEST(`int_list_col`) AS `bfcol_8` WITH OFFSET AS `bfcol_4`
)
ORDER BY
  `bfcol_4` ASC NULLS LAST