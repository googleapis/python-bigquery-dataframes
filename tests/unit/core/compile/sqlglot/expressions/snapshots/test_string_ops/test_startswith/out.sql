WITH `bfcte_0` AS (
  SELECT
    `string_col`
  FROM `bigframes-dev`.`sqlglot_test`.`scalar_types`
)
SELECT
  *,
  STARTS_WITH(`string_col`, 'ab') AS `single`,
  STARTS_WITH(`string_col`, 'ab') OR STARTS_WITH(`string_col`, 'cd') AS `double`,
  FALSE AS `empty`
FROM `bfcte_0`