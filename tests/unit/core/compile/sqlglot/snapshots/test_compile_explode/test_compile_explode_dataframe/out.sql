SELECT
`rowindex` AS `rowindex`,
`rowindex_1` AS `rowindex_1`,
`int_list_col` AS `int_list_col`,
`string_list_col` AS `string_list_col`
FROM
(SELECT
  `t2`.`bfuid_col_2` AS `rowindex`,
  `t2`.`rowindex` AS `rowindex_1`,
  `t2`.`int_list_col`[safe_offset(`t2`.`bfuid_col_796`)] AS `int_list_col`,
  `t2`.`string_list_col`[safe_offset(`t2`.`bfuid_col_796`)] AS `string_list_col`,
  `t2`.`bfuid_col_796` AS `bfuid_col_797`
FROM (
  SELECT
    IF(pos = pos_2, `bfuid_col_796`, NULL) AS `bfuid_col_796`,
    `t1`.`bfuid_col_2`,
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
            ibis_bq_arr_range_czhtci45xzbofjztjmqaunu43q
          FROM UNNEST(generate_array(
            0,
            GREATEST(1, LEAST(ARRAY_LENGTH(`t0`.`int_list_col`), ARRAY_LENGTH(`t0`.`string_list_col`))),
            1
          )) AS ibis_bq_arr_range_czhtci45xzbofjztjmqaunu43q
          WHERE
            ibis_bq_arr_range_czhtci45xzbofjztjmqaunu43q <> GREATEST(1, LEAST(ARRAY_LENGTH(`t0`.`int_list_col`), ARRAY_LENGTH(`t0`.`string_list_col`)))
        ),
        []
      ) AS `bfuid_offset_array_798`,
      `t0`.`rowindex` AS `bfuid_col_2`,
      `t0`.`rowindex`,
      `t0`.`int_list_col`,
      `t0`.`string_list_col`
    FROM (
      SELECT
        `rowindex`,
        `int_list_col`,
        `string_list_col`
      FROM `bigframes-dev.sqlglot_test.repeated_types` FOR SYSTEM_TIME AS OF DATETIME('2025-09-18T23:31:46.924678')
    ) AS `t0`
  ) AS `t1`
  CROSS JOIN UNNEST(GENERATE_ARRAY(0, GREATEST(ARRAY_LENGTH(`t1`.`bfuid_offset_array_798`)) - 1)) AS pos
  CROSS JOIN UNNEST(`t1`.`bfuid_offset_array_798`) AS `bfuid_col_796` WITH OFFSET AS pos_2
  WHERE
    pos = pos_2
    OR (
      pos > (
        ARRAY_LENGTH(`t1`.`bfuid_offset_array_798`) - 1
      )
      AND pos_2 = (
        ARRAY_LENGTH(`t1`.`bfuid_offset_array_798`) - 1
      )
    )
) AS `t2`)
ORDER BY `bfuid_col_797` ASC NULLS LAST