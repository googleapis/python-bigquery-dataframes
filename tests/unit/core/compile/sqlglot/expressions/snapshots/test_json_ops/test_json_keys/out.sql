WITH `bfcte_0` AS (
  SELECT
    `json_col`
  FROM `bigframes-dev`.`sqlglot_test`.`json_types`
)
SELECT
  JSON_KEYS(`json_col`, NULL) AS `json_keys`,
  JSON_KEYS(`json_col`, 2) AS `json_keys_w_max_depth`
FROM `bfcte_0`