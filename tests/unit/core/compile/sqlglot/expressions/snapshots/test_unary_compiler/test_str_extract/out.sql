SELECT
  IF(
    regexp_contains(`t0`.`string_col`, '([a-z]*)'),
    IF(
      1 = 0,
      REGEXP_REPLACE(`t0`.`string_col`, CONCAT('.*?', CONCAT('(', '([a-z]*)', ')'), '.*'), '\\1'),
      REGEXP_REPLACE(`t0`.`string_col`, CONCAT('.*?', '([a-z]*)', '.*'), CONCAT('\\', CAST(1 AS STRING)))
    ),
    NULL
  ) AS `string_col`
FROM (
  SELECT
    `string_col`
  FROM `bigframes-dev.sqlglot_test.scalar_types` FOR SYSTEM_TIME AS OF DATETIME('2025-08-26T20:49:28.159676')
) AS `t0`