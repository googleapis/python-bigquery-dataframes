SELECT
  (
    SELECT
      COALESCE(SUM(__ibis_param_arr_vals__), 0)
    FROM UNNEST(`t0`.`float_list_col`) AS __ibis_param_arr_vals__
  ) AS `sum_float`,
  (
    SELECT
      STDDEV_SAMP(__ibis_param_arr_vals__)
    FROM UNNEST(`t0`.`float_list_col`) AS __ibis_param_arr_vals__
  ) AS `std_float`,
  (
    SELECT
      COUNT(__ibis_param_arr_vals__)
    FROM UNNEST(`t0`.`string_list_col`) AS __ibis_param_arr_vals__
  ) AS `count_str`,
  (
    SELECT
      COALESCE(LOGICAL_OR(__ibis_param_arr_vals__), FALSE)
    FROM UNNEST(`t0`.`bool_list_col`) AS __ibis_param_arr_vals__
  ) AS `any_bool`
FROM `bigframes-dev.sqlglot_test.repeated_types` AS `t0`