SELECT
  NULLIF(
    SUBSTRING(`t0`.`string_col`, IF((
      1 + 1
    ) >= 1, 1 + 1, 1 + 1 + LENGTH(`t0`.`string_col`)), 1),
    ''
  ) AS `string_col`
FROM (
  SELECT
    `string_col`
  FROM `bigframes-dev.sqlglot_test.scalar_types` FOR SYSTEM_TIME AS OF DATETIME('2025-09-30T20:19:48.854671')
) AS `t0`