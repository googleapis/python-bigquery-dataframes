SELECT
`rowindex` AS `rowindex`,
`int_list_col` AS `int_list_col`
FROM
(SELECT
  `t2`.`bfuid_col_14` AS `rowindex`,
  `t2`.`int_list_col`[safe_offset(`t2`.`bfuid_col_862`)] AS `int_list_col`,
  `t2`.`bfuid_col_862` AS `bfuid_col_863`
FROM (
  SELECT
    IF(pos = pos_2, `bfuid_col_862`, NULL) AS `bfuid_col_862`,
    `t1`.`bfuid_col_14`,
    `t1`.`int_list_col`
  FROM (
    SELECT
      IF(
        NOT NULLIF(1, 0) IS NULL
        AND SIGN(1) = SIGN(GREATEST(1, LEAST(ARRAY_LENGTH(`t0`.`int_list_col`))) - 0),
        ARRAY(
          SELECT
            ibis_bq_arr_range_j4jb5nswzzew5gjdx57miqodry
          FROM UNNEST(generate_array(0, GREATEST(1, LEAST(ARRAY_LENGTH(`t0`.`int_list_col`))), 1)) AS ibis_bq_arr_range_j4jb5nswzzew5gjdx57miqodry
          WHERE
            ibis_bq_arr_range_j4jb5nswzzew5gjdx57miqodry <> GREATEST(1, LEAST(ARRAY_LENGTH(`t0`.`int_list_col`)))
        ),
        []
      ) AS `bfuid_offset_array_864`,
      `t0`.`rowindex` AS `bfuid_col_14`,
      `t0`.`int_list_col`
    FROM (
      SELECT
        `rowindex`,
        `int_list_col`
      FROM `bigframes-dev.sqlglot_test.repeated_types` FOR SYSTEM_TIME AS OF DATETIME('2025-09-30T20:19:49.414285')
    ) AS `t0`
  ) AS `t1`
  CROSS JOIN UNNEST(GENERATE_ARRAY(0, GREATEST(ARRAY_LENGTH(`t1`.`bfuid_offset_array_864`)) - 1)) AS pos
  CROSS JOIN UNNEST(`t1`.`bfuid_offset_array_864`) AS `bfuid_col_862` WITH OFFSET AS pos_2
  WHERE
    pos = pos_2
    OR (
      pos > (
        ARRAY_LENGTH(`t1`.`bfuid_offset_array_864`) - 1
      )
      AND pos_2 = (
        ARRAY_LENGTH(`t1`.`bfuid_offset_array_864`) - 1
      )
    )
) AS `t2`)
ORDER BY `bfuid_col_863` ASC NULLS LAST