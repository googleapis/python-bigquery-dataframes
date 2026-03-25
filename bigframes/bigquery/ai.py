# Copyright 2025 Google LLC
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

"""
Integrate BigQuery built-in AI functions into your BigQuery DataFrames workflow.

The ``bigframes.bigquery.ai`` module provides a powerful, Pythonic interface for data scientists
and data engineers to leverage BigQuery ML's Generative AI, Large Language Models (LLMs),
and predictive functions directly on big data via BigQuery DataFrames and Series objects.
These functions enable AI developers to construct scalable MLOps pipelines and perform advanced AI
tasks—such as automated text generation and semantic search—without moving data out of BigQuery's
secure perimeter.

Key capabilities for AI workflows include:

* **Generative AI & LLMs (Gemini):** Use :func:`bigframes.bigquery.ai.generate`
  to orchestrate Gemini models for text analysis, translation, summarization, or
  content generation directly on big data. Specialized versions like
  :func:`~bigframes.bigquery.ai.generate_bool`,
  :func:`~bigframes.bigquery.ai.generate_int`, and
  :func:`~bigframes.bigquery.ai.generate_double` are available for structured
  outputs, perfect for data pipelines.
* **Embeddings & Semantic Search:** Generate vector embeddings for text using
  :func:`~bigframes.bigquery.ai.generate_embedding`. Essential for modern data science,
  enabling robust semantic search and Retrieval-Augmented Generation (RAG) architectures.
* **Classification and Scoring:** Apply robust machine learning models to your data for
  predictive analytics with :func:`~bigframes.bigquery.ai.classify` and
  :func:`~bigframes.bigquery.ai.score`, accelerating the time-to-insight for data analysts.
* **Forecasting:** Predict future values in time-series data using
  :func:`~bigframes.bigquery.ai.forecast` for advanced analytics and business intelligence.

**Example usage:**

    >>> import bigframes.pandas as bpd
    >>> import bigframes.bigquery as bbq

    >>> df = bpd.DataFrame({
    ...     "text_input": [
    ...         "Is this a positive review? The food was terrible.",
    ...     ],
    ... })  # doctest: +SKIP

    >>> # Assuming a Gemini model has been created in BigQuery as 'my_gemini_model'
    >>> result = bq.ai.generate_text("my_gemini_model", df["text_input"])  # doctest: +SKIP

For more information on the underlying BigQuery ML syntax, see:
https://cloud.google.com/bigquery/docs/reference/standard-sql/bigqueryml-syntax-ai-generate-bool
"""

from bigframes.bigquery._operations.ai import (
    classify,
    forecast,
    generate,
    generate_bool,
    generate_double,
    generate_embedding,
    generate_int,
    generate_table,
    generate_text,
    if_,
    score,
)

__all__ = [
    "classify",
    "forecast",
    "generate",
    "generate_bool",
    "generate_double",
    "generate_embedding",
    "generate_int",
    "generate_table",
    "generate_text",
    "if_",
    "score",
]
