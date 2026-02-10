SELECT
  `t1`.`rowindex`,
  `t1`.`int64_col`,
  CASE
    WHEN (
      `t1`.`int64_col`
    ) IS NOT NULL
    THEN IF(
      CAST(CEIL(
        PERCENT_RANK() OVER (
          PARTITION BY (
            `t1`.`int64_col`
          ) IS NOT NULL
          ORDER BY `t1`.`int64_col` IS NULL ASC, `t1`.`int64_col` ASC
        ) * 4
      ) AS INT64) IS NULL,
      CAST(CEIL(
        PERCENT_RANK() OVER (
          PARTITION BY (
            `t1`.`int64_col`
          ) IS NOT NULL
          ORDER BY `t1`.`int64_col` IS NULL ASC, `t1`.`int64_col` ASC
        ) * 4
      ) AS INT64),
      GREATEST(
        1,
        CAST(CEIL(
          PERCENT_RANK() OVER (
            PARTITION BY (
              `t1`.`int64_col`
            ) IS NOT NULL
            ORDER BY `t1`.`int64_col` IS NULL ASC, `t1`.`int64_col` ASC
          ) * 4
        ) AS INT64)
      )
    ) - 1
    ELSE CAST(NULL AS INT64)
  END AS `qcut_w_int`,
  CASE
    WHEN (
      `t1`.`int64_col`
    ) IS NOT NULL
    THEN CASE
      WHEN PERCENT_RANK() OVER (
        PARTITION BY (
          `t1`.`int64_col`
        ) IS NOT NULL
        ORDER BY `t1`.`int64_col` IS NULL ASC, `t1`.`int64_col` ASC
      ) < 0
      THEN NULL
      WHEN PERCENT_RANK() OVER (
        PARTITION BY (
          `t1`.`int64_col`
        ) IS NOT NULL
        ORDER BY `t1`.`int64_col` IS NULL ASC, `t1`.`int64_col` ASC
      ) <= 0.25
      THEN 0
      WHEN PERCENT_RANK() OVER (
        PARTITION BY (
          `t1`.`int64_col`
        ) IS NOT NULL
        ORDER BY `t1`.`int64_col` IS NULL ASC, `t1`.`int64_col` ASC
      ) <= 0.5
      THEN 1
      WHEN PERCENT_RANK() OVER (
        PARTITION BY (
          `t1`.`int64_col`
        ) IS NOT NULL
        ORDER BY `t1`.`int64_col` IS NULL ASC, `t1`.`int64_col` ASC
      ) <= 0.75
      THEN 2
      WHEN PERCENT_RANK() OVER (
        PARTITION BY (
          `t1`.`int64_col`
        ) IS NOT NULL
        ORDER BY `t1`.`int64_col` IS NULL ASC, `t1`.`int64_col` ASC
      ) <= 1
      THEN 3
      ELSE CAST(NULL AS INT64)
    END
    ELSE CAST(NULL AS INT64)
  END AS `qcut_w_list`
FROM (
  SELECT
    `t0`.`int64_col`,
    `t0`.`rowindex`
  FROM `bigframes-dev.sqlglot_test.scalar_types` AS `t0`
) AS `t1`