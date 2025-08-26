SELECT
  ARRAY(
    SELECT
      el
    FROM UNNEST(`t0`.`string_list_col`) AS el WITH OFFSET AS bq_arr_slice
    WHERE
      bq_arr_slice >= IF(1 < 0, ARRAY_LENGTH(`t0`.`string_list_col`) + 1, 1)
      AND bq_arr_slice < IF(5 < 0, ARRAY_LENGTH(`t0`.`string_list_col`) + 5, 5)
  ) AS `string_list_col`
FROM (
  SELECT
    `string_list_col`
  FROM `bigframes-dev.sqlglot_test.repeated_types` FOR SYSTEM_TIME AS OF DATETIME('2025-08-26T20:49:30.548576')
) AS `t0`