"""
This script processes CIViC data, cleans variant names, standardizes them,
and retrieves gene synonyms using the NCBI Datasets API.

Usage:
    python process_civic_script.py input_file output_dir

Parameters:
    - input_file: Path to the input CIViC data file.
    - output_dir: Path to the output directory.
"""

import re
import pandas as pd
import argparse
from typing import List
from ncbi.datasets.openapi import ApiClient as DatasetsApiClient
from ncbi.datasets.openapi import ApiException as DatasetsApiException
from ncbi.datasets import GeneApi as DatasetsGeneApi


def clean_variant(variant):
    """
    Clean and standardize variant names.

    Parameters:
        - variant (str): Variant name.

    Returns:
        - str: Cleaned and standardized variant name.
    """
    variant = re.sub(r'EX(\d+)', r'EXON \1', variant)
    variant = re.sub(r'\bDEL\b', 'DELETION', variant)
    # Remove the "AND" between variants
    variant = re.sub(r'\bAND \b', '', variant)
    pattern = r'([A-Za-z]\d+[A-Za-z])-([A-Za-z]\d+[A-Za-z])'
    cleaned_variant = re.sub(pattern, r'\1 \2', variant)
    return cleaned_variant


def get_civic_genes(civic):
    genes_list = list(civic['gene'].unique())
    return genes_list


def get_genes_synonym(gene_symbols: List[str], output_dir):
    taxon = "human"
    input_dict = {}
    output_list = []

    with DatasetsApiClient() as api_client:
        gene_api = DatasetsGeneApi(api_client)
        try:
            gene_reply = gene_api.gene_metadata_by_tax_and_symbol(gene_symbols, taxon)
            for gene in gene_reply.genes:
                if gene.warnings:
                    continue
                if gene.gene.synonyms:
                    input_dict[gene.gene.symbol] = {'synonyms': gene.gene.synonyms,
                                                    'query': gene.query[0]}
                else:
                    continue
            output_list = [{'gene': details['query'], 'symbol': gene, 'synonym': synonym} for gene, details in input_dict.items() for synonym in details['synonyms']]
        except DatasetsApiException as e:
            print(f"Exception when calling GeneApi: {e}\n")

    df = pd.DataFrame(output_list)
    df.to_csv(f'{output_dir}/gene_synonyms.csv', index=None)


def process_civic_data(civic, output_dir):

    # Converting gene names and variants to uppercase
    civic = civic.assign(gene=civic['gene'].str.upper(),
                         variant=civic['variant'].str.upper())

    # Standardizing variant name
    civic['variant'] = civic['variant'].apply(clean_variant)

    # Saving Processed Data
    civic.to_csv(f"{output_dir}/civic_processed.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process CIViC data")
    parser.add_argument("input_file", help="Path to the input CIViC data file")
    parser.add_argument("output_dir", help="Path to the output directory")
    args = parser.parse_args()

    civic = pd.read_table(args.input_file)
    civic = civic[['gene', 'variant']].drop_duplicates().reset_index(drop=True)

    process_civic_data(civic, args.output_dir)
    get_genes_synonym(get_civic_genes(civic), args.output_dir)
