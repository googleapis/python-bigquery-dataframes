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


def test_multimodal_dataframe():
    # [START bigquery_dataframes_multimodal_dataframe]
    import bigframes

    # Flag to enable the feature
    bigframes.options.experiments.blob = True

    import bigframes.pandas as bpd

    # Create blob columns from wildcard path. The folder contains different file types of image, audio, video, pdf and plain text(null type).
    df_image = bpd.from_glob_path(
        "gs://cloud-samples-data/bigquery/tutorials/cymbal-pets/images/*", name="image"
    )
    # Take only the 5 images to deal with. Preview the content of the Mutimodal DataFrame
    df_image = df_image.head(5)
    df_image
    # [END bigquery_dataframes_multimodal_dataframe]

    # Combine unstructured data with structured data
    df_image["author"] = ["alice", "bob", "bob", "alice", "bob"]  # type: ignore
    df_image["content_type"] = df_image["image"].blob.content_type()
    df_image["size"] = df_image["image"].blob.size()
    df_image["updated"] = df_image["image"].blob.updated()
    df_image

    # filter images and display
    df_image[df_image["author"] == "alice"]["image"].blob.display()

    df_image["blurred"] = df_image["image"].blob.image_blur(
        (8, 8), dst="gs://garrettwu_bucket/image_blur_transformed/"
    )
    df_image["resized"] = df_image["image"].blob.image_resize(
        (300, 200), dst="gs://garrettwu_bucket/image_resize_transformed/"
    )
    df_image["normalized"] = df_image["image"].blob.image_normalize(
        alpha=50.0,
        beta=150.0,
        norm_type="minmax",
        dst="gs://garrettwu_bucket/image_normalize_transformed/",
    )

    # You can also chain functions together
    df_image["blur_resized"] = df_image["blurred"].blob.image_resize(
        (300, 200), dst="gs://garrettwu_bucket/image_blur_resize_transformed/"
    )

    df_image = df_image.head(2)
    from bigframes.ml import llm

    gemini = llm.GeminiTextGenerator(model_name="gemini-1.5-flash-002")

    # Ask the same question on the images
    result = gemini.predict(
        df_image, prompt=["What animal is in ", df_image["image"], "? How many?"]
    )
    result = result[["ml_generate_text_llm_result", "image"]]
    result

    # Ask different questions
    df_image["question"] = [  # type: ignore
        "what animal is it?",
        "How many animals in there? Just answer the exact number.",
    ]
    result = gemini.predict(df_image, prompt=[df_image["question"], df_image["image"]])
    result = result[["ml_generate_text_llm_result", "image"]]
    result

    # Generate embeddings on images
    embed_model = llm.MultimodalEmbeddingGenerator()
    embeddings = embed_model.predict(df_image["image"])
    embeddings

    # PDF chunking
    df_pdf = bpd.from_glob_path(
        "gs://cloud-samples-data/bigquery/tutorials/cymbal-pets/documents/*", name="pdf"
    )
    df_pdf["chunked"] = df_pdf["pdf"].blob.pdf_chunk()
    result = df_pdf["chunked"].explode()
    result
