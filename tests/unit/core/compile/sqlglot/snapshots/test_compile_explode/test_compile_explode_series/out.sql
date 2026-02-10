SELECT
`rowindex` AS `rowindex`,
`int_list_col` AS `int_list_col`
FROM
(SELECT
  `t2`.`bfuid_col_51` AS `rowindex`,
  `t2`.`int_list_col`[safe_offset(`t2`.`bfuid_col_1396`)] AS `int_list_col`,
  `t2`.`bfuid_col_1396` AS `bfuid_col_1397`
FROM (
  SELECT
    IF(pos = pos_2, `bfuid_col_1396`, NULL) AS `bfuid_col_1396`,
    `t1`.`bfuid_col_51`,
    `t1`.`int_list_col`
  FROM (
    SELECT
      IF(
        NOT NULLIF(1, 0) IS NULL
        AND SIGN(1) = SIGN(GREATEST(1, LEAST(ARRAY_LENGTH(`t0`.`int_list_col`))) - 0),
        ARRAY(
          SELECT
            ibis_bq_arr_range_4be4alaztba6vjad4yacu3ibjy
          FROM UNNEST(generate_array(0, GREATEST(1, LEAST(ARRAY_LENGTH(`t0`.`int_list_col`))), 1)) AS ibis_bq_arr_range_4be4alaztba6vjad4yacu3ibjy
          WHERE
            ibis_bq_arr_range_4be4alaztba6vjad4yacu3ibjy <> GREATEST(1, LEAST(ARRAY_LENGTH(`t0`.`int_list_col`)))
        ),
        []
      ) AS `bfuid_offset_array_1398`,
      `t0`.`rowindex` AS `bfuid_col_51`,
      `t0`.`int_list_col`
    FROM `bigframes-dev.sqlglot_test.repeated_types` AS `t0`
  ) AS `t1`
  CROSS JOIN UNNEST(GENERATE_ARRAY(0, GREATEST(ARRAY_LENGTH(`t1`.`bfuid_offset_array_1398`)) - 1)) AS pos
  CROSS JOIN UNNEST(`t1`.`bfuid_offset_array_1398`) AS `bfuid_col_1396` WITH OFFSET AS pos_2
  WHERE
    pos = pos_2
    OR (
      pos > (
        ARRAY_LENGTH(`t1`.`bfuid_offset_array_1398`) - 1
      )
      AND pos_2 = (
        ARRAY_LENGTH(`t1`.`bfuid_offset_array_1398`) - 1
      )
    )
) AS `t2`)
ORDER BY `bfuid_col_1397` ASC NULLS LAST