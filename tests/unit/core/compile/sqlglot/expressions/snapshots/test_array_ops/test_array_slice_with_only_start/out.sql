SELECT
  ARRAY(
    SELECT
      el
    FROM UNNEST(`t0`.`string_list_col`) AS el WITH OFFSET AS bq_arr_slice
    WHERE
      bq_arr_slice >= IF(1 < 0, ARRAY_LENGTH(`t0`.`string_list_col`) + 1, 1)
  ) AS `string_list_col`
FROM (
  SELECT
    `string_list_col`
  FROM `bigframes-dev.sqlglot_test.repeated_types` FOR SYSTEM_TIME AS OF DATETIME('2025-09-30T20:19:49.414285')
) AS `t0`