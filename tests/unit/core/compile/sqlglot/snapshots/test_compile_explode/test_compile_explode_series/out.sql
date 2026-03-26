SELECT `rowindex`, `int_list_col` FROM (SELECT
  `t2`.`bfuid_col_54` AS `rowindex`,
  `t2`.`int_list_col`[safe_offset(`t2`.`bfuid_col_1560`)] AS `int_list_col`,
  `t2`.`bfuid_col_1560` AS `bfuid_col_1561`
FROM (
  SELECT
    IF(pos = pos_2, `bfuid_col_1560`, NULL) AS `bfuid_col_1560`,
    `t1`.`bfuid_col_54`,
    `t1`.`int_list_col`
  FROM (
    SELECT
      IF(
        NOT NULLIF(1, 0) IS NULL
        AND SIGN(1) = SIGN(GREATEST(1, LEAST(ARRAY_LENGTH(`t0`.`int_list_col`))) - 0),
        ARRAY(
          SELECT
            ibis_bq_arr_range_ffjuun3qqfhrzcuj5fnct6bghq
          FROM UNNEST(generate_array(0, GREATEST(1, LEAST(ARRAY_LENGTH(`t0`.`int_list_col`))), 1)) AS ibis_bq_arr_range_ffjuun3qqfhrzcuj5fnct6bghq
          WHERE
            ibis_bq_arr_range_ffjuun3qqfhrzcuj5fnct6bghq <> GREATEST(1, LEAST(ARRAY_LENGTH(`t0`.`int_list_col`)))
        ),
        []
      ) AS `bfuid_offset_array_1562`,
      `t0`.`rowindex` AS `bfuid_col_54`,
      `t0`.`int_list_col`
    FROM `bigframes-dev.sqlglot_test.repeated_types` AS `t0`
  ) AS `t1`
  CROSS JOIN UNNEST(GENERATE_ARRAY(0, GREATEST(ARRAY_LENGTH(`t1`.`bfuid_offset_array_1562`)) - 1)) AS pos
  CROSS JOIN UNNEST(`t1`.`bfuid_offset_array_1562`) AS `bfuid_col_1560` WITH OFFSET AS pos_2
  WHERE
    pos = pos_2
    OR (
      pos > (
        ARRAY_LENGTH(`t1`.`bfuid_offset_array_1562`) - 1
      )
      AND pos_2 = (
        ARRAY_LENGTH(`t1`.`bfuid_offset_array_1562`) - 1
      )
    )
) AS `t2`) AS `t`
ORDER BY `bfuid_col_1561` ASC NULLS LAST