SELECT
FROM (
  SELECT
    `rowindex`,
    `rowindex` AS `rowindex_1`,
    `int_list_col`,
    `string_list_col`
  FROM `bigframes-dev`.`sqlglot_test`.`repeated_types` AS `bft_0`
  LEFT JOIN UNNEST(GENERATE_ARRAY(0, LEAST(ARRAY_LENGTH(`int_list_col`) - 1, ARRAY_LENGTH(`string_list_col`) - 1))) AS `bfcol_13` WITH OFFSET AS `bfcol_7`
)
ORDER BY
  `bfcol_7` ASC NULLS LAST