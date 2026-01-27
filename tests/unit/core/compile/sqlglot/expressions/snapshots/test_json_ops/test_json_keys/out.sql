SELECT
  json_keys(`t1`.`json_col`, NULL) AS `json_keys`,
  json_keys(`t1`.`json_col`, 2) AS `json_keys_w_max_depth`
FROM (
  SELECT
    `t0`.`json_col`
  FROM `bigframes-dev.sqlglot_test.json_types` AS `t0`
) AS `t1`