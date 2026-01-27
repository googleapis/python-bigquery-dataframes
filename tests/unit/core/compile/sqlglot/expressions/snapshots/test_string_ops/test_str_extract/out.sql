SELECT
  IF(
    regexp_contains(`t0`.`string_col`, '([a-z]*)'),
    IF(
      0 = 0,
      REGEXP_REPLACE(`t0`.`string_col`, CONCAT('.*?', CONCAT('(', '([a-z]*)', ')'), '.*'), '\\1'),
      REGEXP_REPLACE(`t0`.`string_col`, CONCAT('.*?', '([a-z]*)', '.*'), CONCAT('\\', CAST(0 AS STRING)))
    ),
    NULL
  ) AS `zero`,
  IF(
    regexp_contains(`t0`.`string_col`, '([a-z]*)'),
    IF(
      1 = 0,
      REGEXP_REPLACE(`t0`.`string_col`, CONCAT('.*?', CONCAT('(', '([a-z]*)', ')'), '.*'), '\\1'),
      REGEXP_REPLACE(`t0`.`string_col`, CONCAT('.*?', '([a-z]*)', '.*'), CONCAT('\\', CAST(1 AS STRING)))
    ),
    NULL
  ) AS `one`
FROM `bigframes-dev.sqlglot_test.scalar_types` AS `t0`