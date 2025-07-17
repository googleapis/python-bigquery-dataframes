SELECT
  `t1`.`rowindex`,
  json_set(json_set(`t1`.`json_col`, '$.a', 100), '$.b', 'hi') AS `json_col`
FROM (
  SELECT
    *
  FROM `bigframes-dev.sqlglot_test.json_types` AS `t0`
) AS `t1`