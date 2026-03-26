SELECT `L_RETURNFLAG`, `L_LINESTATUS`, `SUM_QTY`, `SUM_BASE_PRICE`, `SUM_DISC_PRICE`, `SUM_CHARGE`, `AVG_QTY`, `AVG_PRICE`, `AVG_DISC`, `COUNT_ORDER` FROM (SELECT
  `t2`.`bfuid_col_1891` AS `L_RETURNFLAG`,
  `t2`.`bfuid_col_1892` AS `L_LINESTATUS`,
  `t2`.`bfuid_col_1901` AS `SUM_QTY`,
  `t2`.`bfuid_col_1902` AS `SUM_BASE_PRICE`,
  `t2`.`bfuid_col_1903` AS `SUM_DISC_PRICE`,
  `t2`.`bfuid_col_1904` AS `SUM_CHARGE`,
  `t2`.`bfuid_col_1905` AS `AVG_QTY`,
  `t2`.`bfuid_col_1906` AS `AVG_PRICE`,
  `t2`.`bfuid_col_1907` AS `AVG_DISC`,
  `t2`.`bfuid_col_1908` AS `COUNT_ORDER`
FROM (
  SELECT
    `t1`.`bfuid_col_1891`,
    `t1`.`bfuid_col_1892`,
    COALESCE(SUM(`t1`.`bfuid_col_1887`), 0) AS `bfuid_col_1901`,
    COALESCE(SUM(`t1`.`bfuid_col_1888`), 0) AS `bfuid_col_1902`,
    COALESCE(SUM(`t1`.`bfuid_col_1899`), 0) AS `bfuid_col_1903`,
    COALESCE(SUM(`t1`.`bfuid_col_1900`), 0) AS `bfuid_col_1904`,
    AVG(`t1`.`bfuid_col_1887`) AS `bfuid_col_1905`,
    AVG(`t1`.`bfuid_col_1888`) AS `bfuid_col_1906`,
    AVG(`t1`.`bfuid_col_1889`) AS `bfuid_col_1907`,
    COUNT(`t1`.`bfuid_col_1887`) AS `bfuid_col_1908`
  FROM (
    SELECT
      `t0`.`L_QUANTITY` AS `bfuid_col_1887`,
      `t0`.`L_EXTENDEDPRICE` AS `bfuid_col_1888`,
      `t0`.`L_DISCOUNT` AS `bfuid_col_1889`,
      `t0`.`L_RETURNFLAG` AS `bfuid_col_1891`,
      `t0`.`L_LINESTATUS` AS `bfuid_col_1892`,
      `t0`.`L_EXTENDEDPRICE` * (
        1.0 - `t0`.`L_DISCOUNT`
      ) AS `bfuid_col_1899`,
      (
        `t0`.`L_EXTENDEDPRICE` * (
          1.0 - `t0`.`L_DISCOUNT`
        )
      ) * (
        1.0 + `t0`.`L_TAX`
      ) AS `bfuid_col_1900`
    FROM (
      SELECT
        `L_QUANTITY`,
        `L_EXTENDEDPRICE`,
        `L_DISCOUNT`,
        `L_TAX`,
        `L_RETURNFLAG`,
        `L_LINESTATUS`,
        `L_SHIPDATE`
      FROM `bigframes-dev.tpch.LINEITEM` FOR SYSTEM_TIME AS OF CAST('2026-03-10T18:00:00' AS DATETIME)
    ) AS `t0`
    WHERE
      `t0`.`L_SHIPDATE` <= DATE(1998, 9, 2)
  ) AS `t1`
  GROUP BY
    1,
    2
) AS `t2`
WHERE
  (
    `t2`.`bfuid_col_1891`
  ) IS NOT NULL
  AND (
    `t2`.`bfuid_col_1892`
  ) IS NOT NULL) AS `t`
ORDER BY `L_RETURNFLAG` ASC NULLS LAST ,`L_LINESTATUS` ASC NULLS LAST