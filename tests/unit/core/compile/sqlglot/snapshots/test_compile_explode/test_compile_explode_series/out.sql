SELECT
`rowindex` AS `rowindex`,
`int_list_col` AS `int_list_col`
FROM
(SELECT
  `t2`.`bfuid_col_2` AS `rowindex`,
  `t2`.`int_list_col`[safe_offset(`t2`.`bfuid_col_793`)] AS `int_list_col`,
  `t2`.`bfuid_col_793` AS `bfuid_col_794`
FROM (
  SELECT
    IF(pos = pos_2, `bfuid_col_793`, NULL) AS `bfuid_col_793`,
    `t1`.`bfuid_col_2`,
    `t1`.`int_list_col`
  FROM (
    SELECT
      IF(
        NOT NULLIF(1, 0) IS NULL
        AND SIGN(1) = SIGN(GREATEST(1, LEAST(ARRAY_LENGTH(`t0`.`int_list_col`))) - 0),
        ARRAY(
          SELECT
            ibis_bq_arr_range_7fvof5wapzd5lpnl6kw2tknxqi
          FROM UNNEST(generate_array(0, GREATEST(1, LEAST(ARRAY_LENGTH(`t0`.`int_list_col`))), 1)) AS ibis_bq_arr_range_7fvof5wapzd5lpnl6kw2tknxqi
          WHERE
            ibis_bq_arr_range_7fvof5wapzd5lpnl6kw2tknxqi <> GREATEST(1, LEAST(ARRAY_LENGTH(`t0`.`int_list_col`)))
        ),
        []
      ) AS `bfuid_offset_array_795`,
      `t0`.`rowindex` AS `bfuid_col_2`,
      `t0`.`int_list_col`
    FROM (
      SELECT
        `rowindex`,
        `int_list_col`
      FROM `bigframes-dev.sqlglot_test.repeated_types` FOR SYSTEM_TIME AS OF DATETIME('2025-09-18T23:31:46.924678')
    ) AS `t0`
  ) AS `t1`
  CROSS JOIN UNNEST(GENERATE_ARRAY(0, GREATEST(ARRAY_LENGTH(`t1`.`bfuid_offset_array_795`)) - 1)) AS pos
  CROSS JOIN UNNEST(`t1`.`bfuid_offset_array_795`) AS `bfuid_col_793` WITH OFFSET AS pos_2
  WHERE
    pos = pos_2
    OR (
      pos > (
        ARRAY_LENGTH(`t1`.`bfuid_offset_array_795`) - 1
      )
      AND pos_2 = (
        ARRAY_LENGTH(`t1`.`bfuid_offset_array_795`) - 1
      )
    )
) AS `t2`)
ORDER BY `bfuid_col_794` ASC NULLS LAST