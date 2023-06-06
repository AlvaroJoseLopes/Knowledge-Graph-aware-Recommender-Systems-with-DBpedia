import argparse
import importlib

from data_integration.dataset2class import dataset2class

def get_dataset_class(dataset):
    module_name = f'data_integration.{dataset2class[dataset]["submodule"]}'
    class_name = dataset2class[dataset]['class']
    return module_name, class_name

def main():
    parser = argparse.ArgumentParser(
        description="Script for Data Integration between DBpedia and some standard Recommender System datasets."
    )

    parser.add_argument('-d', '--dataset',  type=str, required=True, help='Choose a supported RS dataset.')
    parser.add_argument('-i', '--input_path', type=str, required=True, help='Path where the dataset is located.')
    parser.add_argument('-o', '--output_path', type=str, required=True, help='Path where the processed dataset will be placed.')
    parser.add_argument('-ci', '--convert_item', action='store_true', help='Use this flag if you want to convert item data.')
    parser.add_argument('-cu', '--convert_user', action='store_true', help='Use this flag if you want to convert user data.')
    parser.add_argument('-cr', '--convert_rating', action='store_true', help='Use this flag if you want to convert rating data.')
    parser.add_argument('-map', '--map_URIs', action='store_true', help='Use this flag if you want to map dataset items with DBpedia.')


    args = parser.parse_args()
    module_name, class_name = get_dataset_class(args.dataset)
    dataset = getattr(importlib.import_module(module_name), class_name)
    dataset = dataset(args.input_path, args.output_path)
    
    if args.convert_item:
        dataset.convert_item_data()
    if args.convert_user:
        dataset.convert_user_data()
    if args.convert_rating:
        dataset.convert_rating_data()
    if args.map_URIs:
        dataset.map_URIs()


if __name__ == '__main__':
    main()
