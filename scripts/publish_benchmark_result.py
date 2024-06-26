import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output_type")
    parser.add_argument("--bigframes_version", default=bigframes.__version__)
    parser.add_argument("--release_version", default="")
    parser.add_argument("--bigquery_table_name")
    args = parser.parse_args()
    df = build_api_coverage_table(args.bigframes_version, args.release_version)

    if args.output_type == "bigquery":
        df.to_gbq(args.bigquery_table_name, if_exists="append")
    elif args.output_type == "docs":
        generate_api_coverage_docs(df)
    elif args.output_type == "summary":
        print_api_coverage_summaries(df)
    else:
        print(f"Unexpected output_type {repr(args.output_type)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
