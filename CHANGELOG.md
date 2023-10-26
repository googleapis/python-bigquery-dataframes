# Changelog

[PyPI History][1]

[1]: https://pypi.org/project/bigframes/#history

## [0.11.0](https://github.com/googleapis/python-bigquery-dataframes/compare/v0.10.0...v0.11.0) (2023-10-26)


### Features

* Add back `reset_session` as an alias for `close_session` ([#124](https://github.com/googleapis/python-bigquery-dataframes/issues/124)) ([694a85a](https://github.com/googleapis/python-bigquery-dataframes/commit/694a85a0ef90d838700014a204d72b23362db1d8))
* Change `query` parameter to `query_or_table` in `read_gbq` ([#127](https://github.com/googleapis/python-bigquery-dataframes/issues/127)) ([f9bb3c4](https://github.com/googleapis/python-bigquery-dataframes/commit/f9bb3c4bc88c5ba2be6f17e12a0ec4f482ce161f))


### Bug Fixes

* Expose `bigframes.pandas.reset_session` as a public API ([#128](https://github.com/googleapis/python-bigquery-dataframes/issues/128)) ([b17e1f4](https://github.com/googleapis/python-bigquery-dataframes/commit/b17e1f43cd0f7567bc5b59b0e916cd20528312b3))
* Use series's own session in series.reindex listlike case ([#135](https://github.com/googleapis/python-bigquery-dataframes/issues/135)) ([95bff3f](https://github.com/googleapis/python-bigquery-dataframes/commit/95bff3f1902bc09dc3310798a42df8ffd31ed8ee))


### Documentation

* Add runnable code samples for DataFrames I/O methods and property ([#129](https://github.com/googleapis/python-bigquery-dataframes/issues/129)) ([6fea8ef](https://github.com/googleapis/python-bigquery-dataframes/commit/6fea8efac35871985677ebeb948a576e64a1ffa4))
* Add runnable code samples for reading methods ([#125](https://github.com/googleapis/python-bigquery-dataframes/issues/125)) ([a669919](https://github.com/googleapis/python-bigquery-dataframes/commit/a669919ff25b56156bd70ccd816a0bf19adb48aa))

## [0.10.0](https://github.com/googleapis/python-bigquery-dataframes/compare/v0.9.0...v0.10.0) (2023-10-19)


### Features

* Implement DataFrame.dot for matrix multiplication ([#67](https://github.com/googleapis/python-bigquery-dataframes/issues/67)) ([29dd414](https://github.com/googleapis/python-bigquery-dataframes/commit/29dd4144c7e0569de3555a16f916be9c4489bf61))

## [0.9.0](https://github.com/googleapis/python-bigquery-dataframes/compare/v0.8.0...v0.9.0) (2023-10-18)


### ⚠ BREAKING CHANGES

* rename `bigframes.pandas.reset_session` to `close_session` ([#101](https://github.com/googleapis/python-bigquery-dataframes/issues/101))

### Features

* Add `bigframes.options.bigquery.application_name` for partner attribution ([#117](https://github.com/googleapis/python-bigquery-dataframes/issues/117)) ([52d64ff](https://github.com/googleapis/python-bigquery-dataframes/commit/52d64ffdbbab16b1d94974b543ce9080be1ec0d1))
* Add AtIndexer getitems ([#107](https://github.com/googleapis/python-bigquery-dataframes/issues/107)) ([752b01f](https://github.com/googleapis/python-bigquery-dataframes/commit/752b01ff9df114c54ed58eb96956e9ce34a8ed47))
* Rename `bigframes.pandas.reset_session` to `close_session` ([#101](https://github.com/googleapis/python-bigquery-dataframes/issues/101)) ([36693bf](https://github.com/googleapis/python-bigquery-dataframes/commit/36693bff398c23e179d9bde95d52cbaddaf85c45))
* Send BigQuery cancel request when canceling bigframes process ([#103](https://github.com/googleapis/python-bigquery-dataframes/issues/103)) ([e325fbb](https://github.com/googleapis/python-bigquery-dataframes/commit/e325fbb1c91e040d87df10f7d4d5ce53f7c052cb))
* Support external packages in `remote_function` ([#98](https://github.com/googleapis/python-bigquery-dataframes/issues/98)) ([ec10c4a](https://github.com/googleapis/python-bigquery-dataframes/commit/ec10c4a5a7833c42e28fe9e7b734bc0c4fb84b6e))
* Use ArrowDtype for STRUCT columns in `to_pandas` ([#85](https://github.com/googleapis/python-bigquery-dataframes/issues/85)) ([9238fad](https://github.com/googleapis/python-bigquery-dataframes/commit/9238fadcfa7e843be6564813ff3131893b79f8b0))


### Bug Fixes

* Support multiindex for three loc getitem overloads ([#113](https://github.com/googleapis/python-bigquery-dataframes/issues/113)) ([68e3cd3](https://github.com/googleapis/python-bigquery-dataframes/commit/68e3cd37258084d045ea1075e5e61df12c28faac))


### Performance Improvements

* If primary keys are defined, `read_gbq` avoids copying table data ([#112](https://github.com/googleapis/python-bigquery-dataframes/issues/112)) ([e6c0cd1](https://github.com/googleapis/python-bigquery-dataframes/commit/e6c0cd1777736e0fa7285da59625fbac487573bd))


### Documentation

* Add documentation for `Series.struct.field` and `Series.struct.explode` ([#114](https://github.com/googleapis/python-bigquery-dataframes/issues/114)) ([a6dab9c](https://github.com/googleapis/python-bigquery-dataframes/commit/a6dab9cdb7dd0e56c93ca96b665ab1be1baac5e5))
* Add open-source link in API doc ([#106](https://github.com/googleapis/python-bigquery-dataframes/issues/106)) ([db51fe3](https://github.com/googleapis/python-bigquery-dataframes/commit/db51fe340f644a0d7c911c11d92c8299a4be3446))
* Update ML overview API doc ([#105](https://github.com/googleapis/python-bigquery-dataframes/issues/105)) ([1b3f3a5](https://github.com/googleapis/python-bigquery-dataframes/commit/1b3f3a5374915b2833c6c1ac05670e9708f07bff))

## [0.8.0](https://github.com/googleapis/python-bigquery-dataframes/compare/v0.7.0...v0.8.0) (2023-10-12)


### ⚠ BREAKING CHANGES

* The default behavior of `to_parquet` is changing from no compression to `'snappy'` compression.

### Features

* Support compression in `to_parquet` ([a8c286f](https://github.com/googleapis/python-bigquery-dataframes/commit/a8c286f0995cc8cf2a4c44fb51855773ecf71f72))


### Bug Fixes

* Create session dataset for remote functions only when needed ([#94](https://github.com/googleapis/python-bigquery-dataframes/issues/94)) ([1d385be](https://github.com/googleapis/python-bigquery-dataframes/commit/1d385be1c68342a66ecb9f28c5efc83c18d0e64c))

## [0.7.0](https://github.com/googleapis/python-bigquery-dataframes/compare/v0.6.0...v0.7.0) (2023-10-11)


### Features

* Add aliases for several series properties ([#80](https://github.com/googleapis/python-bigquery-dataframes/issues/80)) ([c0efec8](https://github.com/googleapis/python-bigquery-dataframes/commit/c0efec8956198247b27904345a795f09c80d3502))
* Add equals methods to series/dataframe ([#76](https://github.com/googleapis/python-bigquery-dataframes/issues/76)) ([636a209](https://github.com/googleapis/python-bigquery-dataframes/commit/636a209e0853501abd50784a11a87cf7f2282ee5))
* Add iat and iloc accessing by tuples of integers ([#90](https://github.com/googleapis/python-bigquery-dataframes/issues/90)) ([228aeba](https://github.com/googleapis/python-bigquery-dataframes/commit/228aeba09782ae2421040c7601c15d4af92790b6))
* Add level param to DataFrame.stack ([#88](https://github.com/googleapis/python-bigquery-dataframes/issues/88)) ([97b8bec](https://github.com/googleapis/python-bigquery-dataframes/commit/97b8bec1175499c74448a4fd46b4888c4b4c35c1))
* Allow df.drop to take an index object ([#68](https://github.com/googleapis/python-bigquery-dataframes/issues/68)) ([740c451](https://github.com/googleapis/python-bigquery-dataframes/commit/740c45176f79d4d2f7f28cb5f6c9eeb1327c8397))
* Use default session connection ([#87](https://github.com/googleapis/python-bigquery-dataframes/issues/87)) ([4ae4ef9](https://github.com/googleapis/python-bigquery-dataframes/commit/4ae4ef995348b95521c4988a8cfb3b5ac792fd69))


### Bug Fixes

* Change the invalid url in docs ([#93](https://github.com/googleapis/python-bigquery-dataframes/issues/93)) ([969800d](https://github.com/googleapis/python-bigquery-dataframes/commit/969800d669204de4d0f2e5e61da521217e55668b))


### Documentation

* Add more preprocessing models into the docs menu. ([#97](https://github.com/googleapis/python-bigquery-dataframes/issues/97)) ([1592315](https://github.com/googleapis/python-bigquery-dataframes/commit/159231505f339173560cd802dae3fed3e63a663b))

## [0.6.0](https://github.com/googleapis/python-bigquery-dataframes/compare/v0.5.0...v0.6.0) (2023-10-04)


### Features

* Add df.unstack ([#63](https://github.com/googleapis/python-bigquery-dataframes/issues/63)) ([4a84714](https://github.com/googleapis/python-bigquery-dataframes/commit/4a84714e2fb07f70c70c79f8b8da9fcb41096e33))
* Add idxmin, idxmax to series, dataframe ([#74](https://github.com/googleapis/python-bigquery-dataframes/issues/74)) ([781307e](https://github.com/googleapis/python-bigquery-dataframes/commit/781307ec22d31a7657f8ee5c6eedc0e419450ccd))
* Add ml.preprocessing.KBinsDiscretizer ([#81](https://github.com/googleapis/python-bigquery-dataframes/issues/81)) ([24c6256](https://github.com/googleapis/python-bigquery-dataframes/commit/24c625638984f6a84191c7a4c8ac9fb6c3cf1dca))
* Add multi-column dataframe merge ([#73](https://github.com/googleapis/python-bigquery-dataframes/issues/73)) ([c9fa85c](https://github.com/googleapis/python-bigquery-dataframes/commit/c9fa85cc338be5e9a8dde59b255690aedbbc1127))
* Add update and align methods to dataframe ([#57](https://github.com/googleapis/python-bigquery-dataframes/issues/57)) ([bf050cf](https://github.com/googleapis/python-bigquery-dataframes/commit/bf050cf475ad8a9e3e0ca3f896ddaf96dbe13ae3))
* Support STRUCT data type with `Series.struct.field` to extract child fields ([#71](https://github.com/googleapis/python-bigquery-dataframes/issues/71)) ([17afac9](https://github.com/googleapis/python-bigquery-dataframes/commit/17afac9ff70a2b93ed70dc7bcce7beb9a53c2ece))


### Bug Fixes

* Avoid `403 response too large to return` error with `read_gbq` and large query results ([#77](https://github.com/googleapis/python-bigquery-dataframes/issues/77)) ([8f3b5b2](https://github.com/googleapis/python-bigquery-dataframes/commit/8f3b5b240f0f28fef92465abc53504e875d7335a))
* Change return type of `Series.loc[scalar]` ([#40](https://github.com/googleapis/python-bigquery-dataframes/issues/40)) ([fff3d45](https://github.com/googleapis/python-bigquery-dataframes/commit/fff3d45f03ffbc7bb23143a1572e3dd157463ca9))
* Fix df/series.iloc by list with multiindex ([#79](https://github.com/googleapis/python-bigquery-dataframes/issues/79)) ([971d091](https://github.com/googleapis/python-bigquery-dataframes/commit/971d091cac9ad662145a3d43d8f9a785eb0ccc23))

## [0.5.0](https://github.com/googleapis/python-bigquery-dataframes/compare/v0.4.0...v0.5.0) (2023-09-28)


### Features

* Add `DataFrame.kurtosis` / `DF.kurt` method ([c1900c2](https://github.com/googleapis/python-bigquery-dataframes/commit/c1900c29a44199d5d8d036d6d842b4f00448fa79))
* Add `DataFrame.rolling` and `DataFrame.expanding` methods ([c1900c2](https://github.com/googleapis/python-bigquery-dataframes/commit/c1900c29a44199d5d8d036d6d842b4f00448fa79))
* Add `items`, `apply` methods to `DataFrame`. ([#43](https://github.com/googleapis/python-bigquery-dataframes/issues/43)) ([3adc1b3](https://github.com/googleapis/python-bigquery-dataframes/commit/3adc1b3aa3e2b218d4fa5debdaa4298276bdf801))
* Add axis param to simple df aggregations ([#52](https://github.com/googleapis/python-bigquery-dataframes/issues/52)) ([9cf9972](https://github.com/googleapis/python-bigquery-dataframes/commit/9cf99721ed83704e6ee28b15c699326c431eb252))
* Add index `dtype`, `astype`, `drop`, `fillna`, aggregate attributes. ([#38](https://github.com/googleapis/python-bigquery-dataframes/issues/38)) ([1a254a4](https://github.com/googleapis/python-bigquery-dataframes/commit/1a254a496633957b9506dd8392dcc6fd10762201))
* Add ml.preprocessing.LabelEncoder ([#50](https://github.com/googleapis/python-bigquery-dataframes/issues/50)) ([2510461](https://github.com/googleapis/python-bigquery-dataframes/commit/25104610e5ffe526315923946533a66713c1d155))
* Add ml.preprocessing.MaxAbsScaler ([#56](https://github.com/googleapis/python-bigquery-dataframes/issues/56)) ([14b262b](https://github.com/googleapis/python-bigquery-dataframes/commit/14b262bde2bb86093bf4df63862e369c5a84b0ad))
* Add ml.preprocessing.MinMaxScaler ([#64](https://github.com/googleapis/python-bigquery-dataframes/issues/64)) ([392113b](https://github.com/googleapis/python-bigquery-dataframes/commit/392113b70d6a8c407accbb6684d75b31261e3741))
* Add more index methods ([#54](https://github.com/googleapis/python-bigquery-dataframes/issues/54)) ([a6e32aa](https://github.com/googleapis/python-bigquery-dataframes/commit/a6e32aa875370063c48ce7922c2aa369a770bd30))
* Support `calculate_p_values` parameter in `bigframes.ml.linear_model.LinearRegression` ([c1900c2](https://github.com/googleapis/python-bigquery-dataframes/commit/c1900c29a44199d5d8d036d6d842b4f00448fa79))
* Support `class_weights="balanced"` in `LogisticRegression` model ([c1900c2](https://github.com/googleapis/python-bigquery-dataframes/commit/c1900c29a44199d5d8d036d6d842b4f00448fa79))
* Support `df[column_name] = df_only_one_column` ([c1900c2](https://github.com/googleapis/python-bigquery-dataframes/commit/c1900c29a44199d5d8d036d6d842b4f00448fa79))
* Support `early_stop` parameter in `bigframes.ml.linear_model.LinearRegression` ([c1900c2](https://github.com/googleapis/python-bigquery-dataframes/commit/c1900c29a44199d5d8d036d6d842b4f00448fa79))
* Support `enable_global_explain` parameter in `bigframes.ml.linear_model.LinearRegression` ([c1900c2](https://github.com/googleapis/python-bigquery-dataframes/commit/c1900c29a44199d5d8d036d6d842b4f00448fa79))
* Support `l2_reg` parameter in `bigframes.ml.linear_model.LinearRegression` ([c1900c2](https://github.com/googleapis/python-bigquery-dataframes/commit/c1900c29a44199d5d8d036d6d842b4f00448fa79))
* Support `learn_rate_strategy` parameter in `bigframes.ml.linear_model.LinearRegression` ([c1900c2](https://github.com/googleapis/python-bigquery-dataframes/commit/c1900c29a44199d5d8d036d6d842b4f00448fa79))
* Support `ls_init_learn_rate` parameter in `bigframes.ml.linear_model.LinearRegression` ([c1900c2](https://github.com/googleapis/python-bigquery-dataframes/commit/c1900c29a44199d5d8d036d6d842b4f00448fa79))
* Support `max_iterations` parameter in `bigframes.ml.linear_model.LinearRegression` ([c1900c2](https://github.com/googleapis/python-bigquery-dataframes/commit/c1900c29a44199d5d8d036d6d842b4f00448fa79))
* Support `min_rel_progress` parameter in `bigframes.ml.linear_model.LinearRegression` ([c1900c2](https://github.com/googleapis/python-bigquery-dataframes/commit/c1900c29a44199d5d8d036d6d842b4f00448fa79))
* Support `optimize_strategy` parameter in `bigframes.ml.linear_model.LinearRegression` ([c1900c2](https://github.com/googleapis/python-bigquery-dataframes/commit/c1900c29a44199d5d8d036d6d842b4f00448fa79))
* Support casting string to integer or float ([#59](https://github.com/googleapis/python-bigquery-dataframes/issues/59)) ([3502f83](https://github.com/googleapis/python-bigquery-dataframes/commit/3502f835b35c437933430698e7a1c9badaddcb99))


### Bug Fixes

* Fix header skipping logic in `read_csv` ([#49](https://github.com/googleapis/python-bigquery-dataframes/issues/49)) ([d56258c](https://github.com/googleapis/python-bigquery-dataframes/commit/d56258cbfcda168cb9e437a021e282818d622d6a))
* Generate unique ids on join to avoid id collisions ([#65](https://github.com/googleapis/python-bigquery-dataframes/issues/65)) ([7ab65e8](https://github.com/googleapis/python-bigquery-dataframes/commit/7ab65e88deb0080e9c36c2709f8a5385ccaf8cf2))
* LabelEncoder params consistent with Sklearn ([#60](https://github.com/googleapis/python-bigquery-dataframes/issues/60)) ([632caec](https://github.com/googleapis/python-bigquery-dataframes/commit/632caec420a7e23188f01b96a00c354d205da74e))
* Loosen filter items tests to accomodate shifting pandas impl ([#41](https://github.com/googleapis/python-bigquery-dataframes/issues/41)) ([edabdbb](https://github.com/googleapis/python-bigquery-dataframes/commit/edabdbb131150707ea9211292cacbb60b8d076dd))


### Performance Improvements

* Add ability to cache dataframe and series to session table ([#51](https://github.com/googleapis/python-bigquery-dataframes/issues/51)) ([416d7cb](https://github.com/googleapis/python-bigquery-dataframes/commit/416d7cb9b560d7e33dcc0227f03a00d43f55ba0d))
* Inline small `Series` and `DataFrames` in query text ([#45](https://github.com/googleapis/python-bigquery-dataframes/issues/45)) ([5e199ec](https://github.com/googleapis/python-bigquery-dataframes/commit/5e199ecf1ecf13a68a2ed0dd4464afd9db977ab1))
* Reimplement unpivot to use cross join rather than union ([#47](https://github.com/googleapis/python-bigquery-dataframes/issues/47)) ([f9a93ce](https://github.com/googleapis/python-bigquery-dataframes/commit/f9a93ce71d053aa17b1e3a2946c90e0227076184))
* Simplify join order to use multiple order keys instead of string. ([#36](https://github.com/googleapis/python-bigquery-dataframes/issues/36)) ([5056da6](https://github.com/googleapis/python-bigquery-dataframes/commit/5056da6b385dbcfc179d2bcbb6549fa539428cda))


### Documentation

* Link to Remote Functions code samples from README and API reference ([c1900c2](https://github.com/googleapis/python-bigquery-dataframes/commit/c1900c29a44199d5d8d036d6d842b4f00448fa79))

## [0.4.0](https://github.com/googleapis/python-bigquery-dataframes/compare/v0.3.2...v0.4.0) (2023-09-16)


### Features

* Add `axis` parameter to `droplevel` and `reorder_levels` ([7c6b0dd](https://github.com/googleapis/python-bigquery-dataframes/commit/7c6b0dd2f99139c8830e762201a45b28486532ff))
* Add `bfill` and `ffill` to `DataFrame` and `Series` ([7c6b0dd](https://github.com/googleapis/python-bigquery-dataframes/commit/7c6b0dd2f99139c8830e762201a45b28486532ff))
* Add `DataFrame.combine` and `DataFrame.combine_first` ([#27](https://github.com/googleapis/python-bigquery-dataframes/issues/27)) ([7c6b0dd](https://github.com/googleapis/python-bigquery-dataframes/commit/7c6b0dd2f99139c8830e762201a45b28486532ff))
* Add `DataFrame.nlargest`, `nsmallest` ([7c6b0dd](https://github.com/googleapis/python-bigquery-dataframes/commit/7c6b0dd2f99139c8830e762201a45b28486532ff))
* Add `DataFrame.pct_change` and `Series.pct_change` ([7c6b0dd](https://github.com/googleapis/python-bigquery-dataframes/commit/7c6b0dd2f99139c8830e762201a45b28486532ff))
* Add `DataFrame.skew` and `GroupBy.skew` ([7c6b0dd](https://github.com/googleapis/python-bigquery-dataframes/commit/7c6b0dd2f99139c8830e762201a45b28486532ff))
* Add `DataFrame.to_dict`, `to_excel`, `to_latex`, `to_records`, `to_string`, `to_markdown`, `to_pickle`, `to_orc` ([7c6b0dd](https://github.com/googleapis/python-bigquery-dataframes/commit/7c6b0dd2f99139c8830e762201a45b28486532ff))
* Add `diff` method to `DataFrame` and `GroupBy` ([7c6b0dd](https://github.com/googleapis/python-bigquery-dataframes/commit/7c6b0dd2f99139c8830e762201a45b28486532ff))
* Add `filter` and `reindex` to `Series` and `DataFrame` ([7c6b0dd](https://github.com/googleapis/python-bigquery-dataframes/commit/7c6b0dd2f99139c8830e762201a45b28486532ff))
* Add `reindex_like` to `DataFrame` and `Series` ([7c6b0dd](https://github.com/googleapis/python-bigquery-dataframes/commit/7c6b0dd2f99139c8830e762201a45b28486532ff))
* Add `swaplevel` to `DataFrame` and `Series` ([7c6b0dd](https://github.com/googleapis/python-bigquery-dataframes/commit/7c6b0dd2f99139c8830e762201a45b28486532ff))
* Add partial support for `Sereies.replace` ([7c6b0dd](https://github.com/googleapis/python-bigquery-dataframes/commit/7c6b0dd2f99139c8830e762201a45b28486532ff))
* Support `DataFrame.loc[bool_series, column] = scalar` ([7c6b0dd](https://github.com/googleapis/python-bigquery-dataframes/commit/7c6b0dd2f99139c8830e762201a45b28486532ff))
* Support a persistent `name` in `remote_function` ([7c6b0dd](https://github.com/googleapis/python-bigquery-dataframes/commit/7c6b0dd2f99139c8830e762201a45b28486532ff))


### Bug Fixes

* `remote_function` uses same credentials as other APIs ([7c6b0dd](https://github.com/googleapis/python-bigquery-dataframes/commit/7c6b0dd2f99139c8830e762201a45b28486532ff))
* Add type hints to models ([7c6b0dd](https://github.com/googleapis/python-bigquery-dataframes/commit/7c6b0dd2f99139c8830e762201a45b28486532ff))
* Raise error when ARIMAPlus is used with Pipeline ([7c6b0dd](https://github.com/googleapis/python-bigquery-dataframes/commit/7c6b0dd2f99139c8830e762201a45b28486532ff))
* Remove `transforms` parameter in `model.fit` (**breaking change**) ([7c6b0dd](https://github.com/googleapis/python-bigquery-dataframes/commit/7c6b0dd2f99139c8830e762201a45b28486532ff))
* Support column joins with "None indexer" ([7c6b0dd](https://github.com/googleapis/python-bigquery-dataframes/commit/7c6b0dd2f99139c8830e762201a45b28486532ff))
* Use for literals `Int64Dtype` in `cut` ([7c6b0dd](https://github.com/googleapis/python-bigquery-dataframes/commit/7c6b0dd2f99139c8830e762201a45b28486532ff))
* Use lowercase strings for parameter literals in `bigframes.ml` (**breaking change**) ([7c6b0dd](https://github.com/googleapis/python-bigquery-dataframes/commit/7c6b0dd2f99139c8830e762201a45b28486532ff))


### Performance Improvements

* `bigframes-api` label to I/O query jobs ([7c6b0dd](https://github.com/googleapis/python-bigquery-dataframes/commit/7c6b0dd2f99139c8830e762201a45b28486532ff))


### Documentation

* Document possible parameter values for PaLM2TextGenerator ([7c6b0dd](https://github.com/googleapis/python-bigquery-dataframes/commit/7c6b0dd2f99139c8830e762201a45b28486532ff))
* Document region logic in README ([7c6b0dd](https://github.com/googleapis/python-bigquery-dataframes/commit/7c6b0dd2f99139c8830e762201a45b28486532ff))
* Fix OneHotEncoder sample ([7c6b0dd](https://github.com/googleapis/python-bigquery-dataframes/commit/7c6b0dd2f99139c8830e762201a45b28486532ff))

## [0.3.2](https://github.com/googleapis/python-bigquery-dataframes/compare/v0.3.1...v0.3.2) (2023-09-06)


### Bug Fixes

* Make release.sh script for PyPI upload executable ([#20](https://github.com/googleapis/python-bigquery-dataframes/issues/20)) ([9951610](https://github.com/googleapis/python-bigquery-dataframes/commit/995161068b118a639903878acfde3202087c25f8))

## [0.3.1](https://github.com/googleapis/python-bigquery-dataframes/compare/v0.3.0...v0.3.1) (2023-09-05)


### Bug Fixes

* **release:** Use correct directory name for release build config ([#17](https://github.com/googleapis/python-bigquery-dataframes/issues/17)) ([3dd25b3](https://github.com/googleapis/python-bigquery-dataframes/commit/3dd25b379ed832ea062e188f483d2789830de67b))

## [0.3.0](https://github.com/googleapis/python-bigquery-dataframes/compare/v0.2.0...v0.3.0) (2023-09-02)


### Features

* Add `bigframes.get_global_session()` and `bigframes.reset_session()` aliases ([a32b747](https://github.com/googleapis/python-bigquery-dataframes/commit/a32b74751785c8e8aec40ce01df639dd7c4fbb77))
* Add `bigframes.pandas.read_pickle` function ([a32b747](https://github.com/googleapis/python-bigquery-dataframes/commit/a32b74751785c8e8aec40ce01df639dd7c4fbb77))
* Add `components_`, `explained_variance_`, and `explained_variance_ratio_` properties to `bigframes.ml.decomposition.PCA` ([89b9503](https://github.com/googleapis/python-bigquery-dataframes/commit/89b95033d6b449bfc21249057d7c024d096c80d0))
* Add `fit_transform` to `bigquery.ml` transformers ([a32b747](https://github.com/googleapis/python-bigquery-dataframes/commit/a32b74751785c8e8aec40ce01df639dd7c4fbb77))
* Add `Series.dropna` and `DataFrame.fillna` ([8fab755](https://github.com/googleapis/python-bigquery-dataframes/commit/8fab75576757230bca5c7df10994837ac406300f))
* Add `Series.str` methods `isalpha`, `isdigit`, `isdecimal`, `isalnum`, `isspace`, `islower`, `isupper`, `zfill`, `center` ([a32b747](https://github.com/googleapis/python-bigquery-dataframes/commit/a32b74751785c8e8aec40ce01df639dd7c4fbb77))
* Support `bigframes.pandas.merge()` ([8fab755](https://github.com/googleapis/python-bigquery-dataframes/commit/8fab75576757230bca5c7df10994837ac406300f))
* Support `DataFrame.isin` with list and dict inputs ([8fab755](https://github.com/googleapis/python-bigquery-dataframes/commit/8fab75576757230bca5c7df10994837ac406300f))
* Support `DataFrame.pivot` ([a32b747](https://github.com/googleapis/python-bigquery-dataframes/commit/a32b74751785c8e8aec40ce01df639dd7c4fbb77))
* Support `DataFrame.stack` ([89b9503](https://github.com/googleapis/python-bigquery-dataframes/commit/89b95033d6b449bfc21249057d7c024d096c80d0))
* Support `DataFrame`-`DataFrame` binary operations ([8fab755](https://github.com/googleapis/python-bigquery-dataframes/commit/8fab75576757230bca5c7df10994837ac406300f))
* Support `df[my_column] = [a python list]` ([89b9503](https://github.com/googleapis/python-bigquery-dataframes/commit/89b95033d6b449bfc21249057d7c024d096c80d0))
* Support `Index.is_monotonic` ([8fab755](https://github.com/googleapis/python-bigquery-dataframes/commit/8fab75576757230bca5c7df10994837ac406300f))
* Support `np.arcsin`, `np.arccos`, `np.arctan`, `np.sinh`, `np.cosh`, `np.tanh`, `np.arcsinh`, `np.arccosh`, `np.arctanh`, `np.exp` with Series argument ([89b9503](https://github.com/googleapis/python-bigquery-dataframes/commit/89b95033d6b449bfc21249057d7c024d096c80d0))
* Support `np.sin`, `np.cos`, `np.tan`, `np.log`, `np.log10`, `np.sqrt`, `np.abs` with Series argument ([89b9503](https://github.com/googleapis/python-bigquery-dataframes/commit/89b95033d6b449bfc21249057d7c024d096c80d0))
* Support `pow()` and power operator in `DataFrame` and `Series` ([8fab755](https://github.com/googleapis/python-bigquery-dataframes/commit/8fab75576757230bca5c7df10994837ac406300f))
* Support `read_json` with `engine=bigquery` for newline-delimited JSON files ([89b9503](https://github.com/googleapis/python-bigquery-dataframes/commit/89b95033d6b449bfc21249057d7c024d096c80d0))
* Support `Series.corr` ([89b9503](https://github.com/googleapis/python-bigquery-dataframes/commit/89b95033d6b449bfc21249057d7c024d096c80d0))
* Support `Series.map` ([8fab755](https://github.com/googleapis/python-bigquery-dataframes/commit/8fab75576757230bca5c7df10994837ac406300f))
* Support for `np.add`, `np.subtract`, `np.multiply`, `np.divide`, `np.power` ([8fab755](https://github.com/googleapis/python-bigquery-dataframes/commit/8fab75576757230bca5c7df10994837ac406300f))
* Support MultiIndex for DataFrame columns ([a32b747](https://github.com/googleapis/python-bigquery-dataframes/commit/a32b74751785c8e8aec40ce01df639dd7c4fbb77))
* Use `pandas.Index` for column labels ([a32b747](https://github.com/googleapis/python-bigquery-dataframes/commit/a32b74751785c8e8aec40ce01df639dd7c4fbb77))
* Use default session and connection in `ml.llm` and `ml.imported` ([8fab755](https://github.com/googleapis/python-bigquery-dataframes/commit/8fab75576757230bca5c7df10994837ac406300f))


### Bug Fixes

* Add error message to `set_index` ([a32b747](https://github.com/googleapis/python-bigquery-dataframes/commit/a32b74751785c8e8aec40ce01df639dd7c4fbb77))
* Align column names with pandas in `DataFrame.agg` results ([89b9503](https://github.com/googleapis/python-bigquery-dataframes/commit/89b95033d6b449bfc21249057d7c024d096c80d0))
* Allow (but still not recommended) `ORDER BY` in `read_gbq` input when an `index_col` is defined ([89b9503](https://github.com/googleapis/python-bigquery-dataframes/commit/89b95033d6b449bfc21249057d7c024d096c80d0))
* Check for IAM role on the BigQuery connection when initializing a `remote_function` ([89b9503](https://github.com/googleapis/python-bigquery-dataframes/commit/89b95033d6b449bfc21249057d7c024d096c80d0))
* Check that types are specified in `read_gbq_function` ([a32b747](https://github.com/googleapis/python-bigquery-dataframes/commit/a32b74751785c8e8aec40ce01df639dd7c4fbb77))
* Don't use query cache for Session construction ([a32b747](https://github.com/googleapis/python-bigquery-dataframes/commit/a32b74751785c8e8aec40ce01df639dd7c4fbb77))
* Include survey link in abstract `NotImplementedError` exception messages ([89b9503](https://github.com/googleapis/python-bigquery-dataframes/commit/89b95033d6b449bfc21249057d7c024d096c80d0))
* Label temp table creation jobs with `source=bigquery-dataframes-temp` label ([89b9503](https://github.com/googleapis/python-bigquery-dataframes/commit/89b95033d6b449bfc21249057d7c024d096c80d0))
* Make `X_train` argument names consistent across methods ([8fab755](https://github.com/googleapis/python-bigquery-dataframes/commit/8fab75576757230bca5c7df10994837ac406300f))
* Raise AttributeError for unimplemented pandas methods ([89b9503](https://github.com/googleapis/python-bigquery-dataframes/commit/89b95033d6b449bfc21249057d7c024d096c80d0))
* Raise exception for invalid function in `read_gbq_function` ([a32b747](https://github.com/googleapis/python-bigquery-dataframes/commit/a32b74751785c8e8aec40ce01df639dd7c4fbb77))
* Support spaces in column names in `DataFrame` initializater ([89b9503](https://github.com/googleapis/python-bigquery-dataframes/commit/89b95033d6b449bfc21249057d7c024d096c80d0))


### Performance Improvements

* Add local cache for `__repr_*__` methods ([a32b747](https://github.com/googleapis/python-bigquery-dataframes/commit/a32b74751785c8e8aec40ce01df639dd7c4fbb77))
* Lazily instantiate client library objects ([89b9503](https://github.com/googleapis/python-bigquery-dataframes/commit/89b95033d6b449bfc21249057d7c024d096c80d0))
* Use `row_number()` filter for `head` / `tail` ([8fab755](https://github.com/googleapis/python-bigquery-dataframes/commit/8fab75576757230bca5c7df10994837ac406300f))


### Documentation

* Add ML section under Overview ([a32b747](https://github.com/googleapis/python-bigquery-dataframes/commit/a32b74751785c8e8aec40ce01df639dd7c4fbb77))
* Add release status to table of contents ([a32b747](https://github.com/googleapis/python-bigquery-dataframes/commit/a32b74751785c8e8aec40ce01df639dd7c4fbb77))
* Add samples and best practices to `read_gbq` docs ([a32b747](https://github.com/googleapis/python-bigquery-dataframes/commit/a32b74751785c8e8aec40ce01df639dd7c4fbb77))
* Correct the return types of Dataframe and Series ([a32b747](https://github.com/googleapis/python-bigquery-dataframes/commit/a32b74751785c8e8aec40ce01df639dd7c4fbb77))
* Create subfolders for notebooks ([a32b747](https://github.com/googleapis/python-bigquery-dataframes/commit/a32b74751785c8e8aec40ce01df639dd7c4fbb77))
* Fix link to GitHub ([89b9503](https://github.com/googleapis/python-bigquery-dataframes/commit/89b95033d6b449bfc21249057d7c024d096c80d0))
* Highlight bigframes is open-source ([a32b747](https://github.com/googleapis/python-bigquery-dataframes/commit/a32b74751785c8e8aec40ce01df639dd7c4fbb77))
* Sample ML Drug Name Generation notebook ([a32b747](https://github.com/googleapis/python-bigquery-dataframes/commit/a32b74751785c8e8aec40ce01df639dd7c4fbb77))
* Set `options.bigquery.project` in sample code ([89b9503](https://github.com/googleapis/python-bigquery-dataframes/commit/89b95033d6b449bfc21249057d7c024d096c80d0))
* Transform remote function user guide into sample code ([a32b747](https://github.com/googleapis/python-bigquery-dataframes/commit/a32b74751785c8e8aec40ce01df639dd7c4fbb77))
* Update remote function notebook with read_gbq_function usage ([8fab755](https://github.com/googleapis/python-bigquery-dataframes/commit/8fab75576757230bca5c7df10994837ac406300f))

## 0.2.0 (2023-08-17)

### Features
* Add KMeans.cluster_centers_.
* Allow column labels to be any type handled by bq df, column labels can be integers now.
* Add dataframegroupby.agg().
* Add Series Property is_monotonic_increasing and is_monotonic_decreasing.
* Add match, fullmatch, get, pad str methods.
* Add series isin function.

### Bug Fixes
* Update ML package to use sessions for queries.
* Optimize `read_gbq` with `index_col` set to cluster by `index_col`.
* Raise ValueError if the location mismatched.
* `read_gbq` no longer uses 'time travel' with query inputs.

### Documentation
* Add docstring to _uniform_sampling to avoid user using it.

## 0.1.1 (2023-08-14)

### Documentation

* Correct link to code repository in `setup.py` and use correct terminology for
  `console.cloud.google.com` links.

## 0.1.0 (2023-08-11)

### Features

* Add `bigframes.pandas` package with an API compatible with
  [pandas](https://pandas.pydata.org/). Supported data sources include:
  BigQuery SQL queries, BigQuery tables, CSV (local and GCS), Parquet (local
  and Cloud Storage), and more.
* Add `bigframes.ml` package with an API inspired by
  [scikit-learn](https://scikit-learn.org/stable/). Train machine learning
  models and run batch predicition, powered by [BigQuery
  ML](https://cloud.google.com/bigquery/docs/bqml-introduction).

## [0.0.0](https://pypi.org/project/bigframes/0.0.0/) (2023-02-22)

* Empty package to reserve package name.
