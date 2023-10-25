
# Copyright 2023 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


def test_bqml_getting_started():
    #[START bigquery_getting_Started_bqml_tutorial]
    #DataFrame created from a BigQuery table:
    import bigframes.pandas as bpd 
    import bigframes 

    # Original sql query from tutorial, translated to Python using BigQuery BigFrames dataframes 
    df = bpd.read_gbq('''
    SELECT GENERATE_UUID() AS rowindex, *
    FROM
    `bigquery-public-data.google_analytics_sample.ga_sessions_*`
    WHERE
    _TABLE_SUFFIX BETWEEN '20160801' AND '20170630'
    ''',
    index_col='rowindex')

    #Printing dataframe, setting totals value 
    totals = df['totals']

    #Using totals, selecting id for transaction example 
    totals['0000fb2c-2861-40be-9c6c-309afd7e7883']
    transactions = totals.struct.field("transactions")
    #Columns to indicate whether there was purchase 
    label = transactions.notnull().map({True: 1, False: 0})

    #Operating systems of users, extracting child fields of a struct as a Series
    operatingSystem = df['device'].struct.field("operatingSystem")
    operatingSystem = operatingSystem.fillna("")

    #Indicates whether the users devices are mobile 
    isMobile = df['device'].struct.field("isMobile")

    #Country from which the sessions originate, IP address based 
    country = df['geoNetwork'].struct.field("country").fillna("")

    #Total number of pageviews within the session,
    pageviews = totals.struct.field("pageviews").fillna(0)

    #Setting features for dataframe, 
    features = bpd.DataFrame({
        'os': operatingSystem,
        'is_mobile': isMobile,
        'pageviews': pageviews
    })

    #Printing out the dataframe 
    df 
    
    #Creating a logistics regression model - 
    from bigframes.ml.linear_model import LogisticRegression
    model = LogisticRegression()
    #Model training parameters, 
    model.fit(features, label)
    #Write a DataFRame to a BigQuery table- 
    model.to_gbq('bqml_tutorial.sample_model', replace = True)
