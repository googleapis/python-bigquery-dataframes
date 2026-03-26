SELECT
  IF(
    SUBSTRING(
      `t0`.`string_col`,
      IF(
        (
          IF(1 >= 0, 1, GREATEST(0, LENGTH(`t0`.`string_col`) + 1)) + 1
        ) >= 1,
        IF(1 >= 0, 1, GREATEST(0, LENGTH(`t0`.`string_col`) + 1)) + 1,
        IF(1 >= 0, 1, GREATEST(0, LENGTH(`t0`.`string_col`) + 1)) + 1 + LENGTH(`t0`.`string_col`)
      ),
      GREATEST(
        0,
        IF(5 >= 0, 5, GREATEST(0, LENGTH(`t0`.`string_col`) + 5)) - IF(1 >= 0, 1, GREATEST(0, LENGTH(`t0`.`string_col`) + 1))
      )
    ) <> '',
    SUBSTRING(
      `t0`.`string_col`,
      IF(
        (
          IF(1 >= 0, 1, GREATEST(0, LENGTH(`t0`.`string_col`) + 1)) + 1
        ) >= 1,
        IF(1 >= 0, 1, GREATEST(0, LENGTH(`t0`.`string_col`) + 1)) + 1,
        IF(1 >= 0, 1, GREATEST(0, LENGTH(`t0`.`string_col`) + 1)) + 1 + LENGTH(`t0`.`string_col`)
      ),
      GREATEST(
        0,
        IF(5 >= 0, 5, GREATEST(0, LENGTH(`t0`.`string_col`) + 5)) - IF(1 >= 0, 1, GREATEST(0, LENGTH(`t0`.`string_col`) + 1))
      )
    ),
    NULL
  ) AS `string_slice`,
  ARRAY(
    SELECT
      el
    FROM UNNEST([`t0`.`int64_col`, `t0`.`int64_too`]) AS el WITH OFFSET AS bq_arr_slice
    WHERE
      bq_arr_slice >= IF(1 < 0, ARRAY_LENGTH([`t0`.`int64_col`, `t0`.`int64_too`]) + 1, 1)
  ) AS `slice_only_start`,
  ARRAY(
    SELECT
      el
    FROM UNNEST([`t0`.`int64_col`, `t0`.`int64_too`]) AS el WITH OFFSET AS bq_arr_slice
    WHERE
      bq_arr_slice >= IF(1 < 0, ARRAY_LENGTH([`t0`.`int64_col`, `t0`.`int64_too`]) + 1, 1)
      AND bq_arr_slice < IF(5 < 0, ARRAY_LENGTH([`t0`.`int64_col`, `t0`.`int64_too`]) + 5, 5)
  ) AS `slice_start_stop`
FROM `bigframes-dev.sqlglot_test.scalar_types` AS `t0`