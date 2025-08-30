SELECT
`rowindex` AS `rowindex`,
`int_list_col` AS `int_list_col`
FROM
(SELECT
  `t2`.`bfuid_col_655` AS `rowindex`,
  `t2`.`int_list_col`[safe_offset(`t2`.`bfuid_col_776`)] AS `int_list_col`,
  `t2`.`bfuid_col_776` AS `bfuid_col_777`
FROM (
  SELECT
    IF(pos = pos_2, `bfuid_col_776`, NULL) AS `bfuid_col_776`,
    `t1`.`bfuid_col_655`,
    `t1`.`int_list_col`
  FROM (
    SELECT
      IF(
        NOT NULLIF(1, 0) IS NULL
        AND SIGN(1) = SIGN(GREATEST(1, LEAST(ARRAY_LENGTH(`t0`.`int_list_col`))) - 0),
        ARRAY(
          SELECT
            ibis_bq_arr_range_qur3yk7lmfbexjnriqyy72jdyy
          FROM UNNEST(generate_array(0, GREATEST(1, LEAST(ARRAY_LENGTH(`t0`.`int_list_col`))), 1)) AS ibis_bq_arr_range_qur3yk7lmfbexjnriqyy72jdyy
          WHERE
            ibis_bq_arr_range_qur3yk7lmfbexjnriqyy72jdyy <> GREATEST(1, LEAST(ARRAY_LENGTH(`t0`.`int_list_col`)))
        ),
        []
      ) AS `bfuid_offset_array_778`,
      `t0`.`rowindex` AS `bfuid_col_655`,
      `t0`.`int_list_col`
    FROM (
      SELECT
        `rowindex`,
        `int_list_col`
      FROM `bigframes-dev.sqlglot_test.repeated_types` FOR SYSTEM_TIME AS OF DATETIME('2025-08-26T20:49:30.548576')
    ) AS `t0`
  ) AS `t1`
  CROSS JOIN UNNEST(GENERATE_ARRAY(0, GREATEST(ARRAY_LENGTH(`t1`.`bfuid_offset_array_778`)) - 1)) AS pos
  CROSS JOIN UNNEST(`t1`.`bfuid_offset_array_778`) AS `bfuid_col_776` WITH OFFSET AS pos_2
  WHERE
    pos = pos_2
    OR (
      pos > (
        ARRAY_LENGTH(`t1`.`bfuid_offset_array_778`) - 1
      )
      AND pos_2 = (
        ARRAY_LENGTH(`t1`.`bfuid_offset_array_778`) - 1
      )
    )
) AS `t2`)
ORDER BY `bfuid_col_777` ASC NULLS LAST