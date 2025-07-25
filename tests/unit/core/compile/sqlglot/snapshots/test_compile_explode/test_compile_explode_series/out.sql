SELECT
`rowindex` AS `rowindex`,
`int_list_col` AS `int_list_col`
FROM
(SELECT
  `t2`.`bfuid_col_23` AS `rowindex`,
  `t2`.`int_list_col`[safe_offset(`t2`.`bfuid_col_77`)] AS `int_list_col`,
  `t2`.`bfuid_col_77` AS `bfuid_col_78`
FROM (
  SELECT
    IF(pos = pos_2, `bfuid_col_77`, NULL) AS `bfuid_col_77`,
    `t1`.`bfuid_col_23`,
    `t1`.`int_list_col`
  FROM (
    SELECT
      IF(
        NOT NULLIF(1, 0) IS NULL
        AND SIGN(1) = SIGN(GREATEST(1, LEAST(ARRAY_LENGTH(`t0`.`int_list_col`))) - 0),
        ARRAY(
          SELECT
            ibis_bq_arr_range_73lemazenjcffbrmk4yifhi5ru
          FROM UNNEST(generate_array(0, GREATEST(1, LEAST(ARRAY_LENGTH(`t0`.`int_list_col`))), 1)) AS ibis_bq_arr_range_73lemazenjcffbrmk4yifhi5ru
          WHERE
            ibis_bq_arr_range_73lemazenjcffbrmk4yifhi5ru <> GREATEST(1, LEAST(ARRAY_LENGTH(`t0`.`int_list_col`)))
        ),
        []
      ) AS `bfuid_offset_array_79`,
      `t0`.`rowindex` AS `bfuid_col_23`,
      `t0`.`int_list_col`
    FROM `bigframes-dev.sqlglot_test.repeated_types` AS `t0`
  ) AS `t1`
  CROSS JOIN UNNEST(GENERATE_ARRAY(0, GREATEST(ARRAY_LENGTH(`t1`.`bfuid_offset_array_79`)) - 1)) AS pos
  CROSS JOIN UNNEST(`t1`.`bfuid_offset_array_79`) AS `bfuid_col_77` WITH OFFSET AS pos_2
  WHERE
    pos = pos_2
    OR (
      pos > (
        ARRAY_LENGTH(`t1`.`bfuid_offset_array_79`) - 1
      )
      AND pos_2 = (
        ARRAY_LENGTH(`t1`.`bfuid_offset_array_79`) - 1
      )
    )
) AS `t2`)
ORDER BY `bfuid_col_78` ASC NULLS LAST