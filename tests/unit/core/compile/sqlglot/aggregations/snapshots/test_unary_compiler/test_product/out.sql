SELECT
  *
FROM (
  SELECT
    CASE
      WHEN LOGICAL_OR(`t1`.`int64_col` = 0)
      THEN 0
      ELSE (
        POWER(
          2,
          SUM(CASE WHEN `t1`.`int64_col` = 0 THEN 0 ELSE LOG(ABS(`t1`.`int64_col`), 2) END)
        )
      ) * (
        POWER(-1, MOD(SUM(CASE WHEN SIGN(`t1`.`int64_col`) = -1 THEN 1 ELSE 0 END), 2))
      )
    END AS `int64_col`
  FROM (
    SELECT
      `t0`.`int64_col`
    FROM `bigframes-dev.sqlglot_test.scalar_types` AS `t0`
  ) AS `t1`
) AS `t2`