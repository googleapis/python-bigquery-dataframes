SELECT
  *
FROM (
  SELECT
    COALESCE(
      string_agg(
        `t1`.`string_col` ORDER BY (`t1`.`string_col` IS NULL) ASC, (`t1`.`string_col`) ASC,
        ','
      ),
      ''
    ) AS `string_col`
  FROM (
    SELECT
      `t0`.`string_col`
    FROM `bigframes-dev.sqlglot_test.scalar_types` AS `t0`
  ) AS `t1`
) AS `t2`