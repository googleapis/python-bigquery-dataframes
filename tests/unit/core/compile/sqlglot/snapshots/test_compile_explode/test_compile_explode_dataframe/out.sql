SELECT
`rowindex` AS `rowindex`,
`rowindex_1` AS `rowindex_1`,
`int_list_col` AS `int_list_col`,
`string_list_col` AS `string_list_col`
FROM
(SELECT
  `t2`.`bfuid_col_251` AS `rowindex`,
  `t2`.`rowindex` AS `rowindex_1`,
  `t2`.`int_list_col`[safe_offset(`t2`.`bfuid_col_351`)] AS `int_list_col`,
  `t2`.`string_list_col`[safe_offset(`t2`.`bfuid_col_351`)] AS `string_list_col`,
  `t2`.`bfuid_col_351` AS `bfuid_col_352`
FROM (
  SELECT
    IF(pos = pos_2, `bfuid_col_351`, NULL) AS `bfuid_col_351`,
    `t1`.`bfuid_col_251`,
    `t1`.`rowindex`,
    `t1`.`int_list_col`,
    `t1`.`string_list_col`
  FROM (
    SELECT
      IF(
        NOT NULLIF(1, 0) IS NULL
        AND SIGN(1) = SIGN(
          GREATEST(1, LEAST(ARRAY_LENGTH(`t0`.`int_list_col`), ARRAY_LENGTH(`t0`.`string_list_col`))) - 0
        ),
        ARRAY(
          SELECT
            ibis_bq_arr_range_wexsb7vmnrdu3deb7fbv2elode
          FROM UNNEST(generate_array(
            0,
            GREATEST(1, LEAST(ARRAY_LENGTH(`t0`.`int_list_col`), ARRAY_LENGTH(`t0`.`string_list_col`))),
            1
          )) AS ibis_bq_arr_range_wexsb7vmnrdu3deb7fbv2elode
          WHERE
            ibis_bq_arr_range_wexsb7vmnrdu3deb7fbv2elode <> GREATEST(1, LEAST(ARRAY_LENGTH(`t0`.`int_list_col`), ARRAY_LENGTH(`t0`.`string_list_col`)))
        ),
        []
      ) AS `bfuid_offset_array_353`,
      `t0`.`rowindex` AS `bfuid_col_251`,
      `t0`.`rowindex`,
      `t0`.`int_list_col`,
      `t0`.`string_list_col`
    FROM (
      SELECT
        `rowindex`,
        `int_list_col`,
        `string_list_col`
      FROM `bigframes-dev.sqlglot_test.repeated_types` FOR SYSTEM_TIME AS OF DATETIME('2025-08-15T22:12:37.046629')
    ) AS `t0`
  ) AS `t1`
  CROSS JOIN UNNEST(GENERATE_ARRAY(0, GREATEST(ARRAY_LENGTH(`t1`.`bfuid_offset_array_353`)) - 1)) AS pos
  CROSS JOIN UNNEST(`t1`.`bfuid_offset_array_353`) AS `bfuid_col_351` WITH OFFSET AS pos_2
  WHERE
    pos = pos_2
    OR (
      pos > (
        ARRAY_LENGTH(`t1`.`bfuid_offset_array_353`) - 1
      )
      AND pos_2 = (
        ARRAY_LENGTH(`t1`.`bfuid_offset_array_353`) - 1
      )
    )
) AS `t2`)
ORDER BY `bfuid_col_352` ASC NULLS LAST