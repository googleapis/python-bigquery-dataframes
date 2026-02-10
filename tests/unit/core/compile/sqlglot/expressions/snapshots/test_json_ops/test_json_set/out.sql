SELECT
  JSON_SET(`t1`.`json_col`, '$.a', 100) AS `json_col`
FROM (
  SELECT
    `t0`.`json_col`
  FROM `bigframes-dev.sqlglot_test.json_types` AS `t0`
) AS `t1`