import os
import queue
import string
import pandas as pd
from string import Template
from .worker import Worker

from SPARQLWrapper import SPARQLWrapper, JSON
from tqdm import tqdm

class Dataset():
    def __init__(self, input_path, output_path, n_workers=1):
        r"""
        Base class for Recommender System's datasets
        """
        self.input_path = input_path
        self.output_path = output_path
        self.n_workers = n_workers

        self.sparql_endpoint = "http://dbpedia.org/sparql"
        self.timeout = 180

        # Output files
        self.item_filename = os.path.join(self.output_path, 'item.csv')
        self.user_filename = os.path.join(self.output_path, 'user.csv')
        self.rating_filename = os.path.join(self.output_path, 'rating.csv')
        self.map_filename = os.path.join(self.output_path, 'map.csv')
        self.enriched_filename = os.path.join(self.output_path, 'enriched.csv')

        # create output path if doesn't exist
        self.has_output_path()
        # Mapping regex special chars
        self._special_chars_map = str.maketrans({x:'' for x in string.punctuation})


        # The following variables need to be overwritten by the subclasses of Dataset()
        self.dataset_name = ''
        self.item_separator = '' 
        self.user_separator = ''
        self.rating_separator = ''
        # Chosen features to be extracted from dataset
        self.item_features = []
        self.user_features = []
        self.rating_features = []
        self.map_query_template = Template(None)
        self.enrich_query_template =  Template(None)


    def has_output_path(self):
        if not os.path.isdir(self.output_path):
            os.mkdir(self.output_path)
    
    def load_item_data(self) -> pd.DataFrame():
        """
        Loads item info of Dataset
        :return: returns pd.Dataframe() containing each item info.
        """
        raise NotImplementedError

    def load_user_data(self) -> pd.DataFrame():
        """
        Loads user info of Dataset
        :return: returns pd.Dataframe() containing each user info.
        """
        raise NotImplementedError

    def load_rating_data(self) -> pd.DataFrame():
        """
        Loads rating interactions of Dataset
        :return: returns pd.Dataframe() containing each rating interaction.
        """
        raise NotImplementedError
    
    def entity_linking(self, df_item) -> pd.DataFrame():
        """
        Entity link each item to their corresponding DBpedia's URI
        :return: returns pd.Dataframe() containing each item_id and their mapped URI 
        """
        raise NotImplementedError

    def enrich(self, df_map):
        raise NotImplementedError
    
    def get_map_query(self, *args, **kwargs) -> str:
        """
        Returns query parameters to be substituted in the map query template.
        :returns: returns string query
        """
        raise NotImplementedError

    def get_enrich_query(self, *args, **kwargs) -> str:
        """
        Returns query parameters to be substituted in the enrich query template.
        :returns: returns string query
        """
        raise NotImplementedError

    def parallel_queries(self, queue, return_type=JSON):
        """
        Parallel query SPARQL endpoint using Threads
        :arguments: 
            queue: queue of tuples indicating the item_id and SPARQL query string
        :returns: list containing tuples with item_id and response result in JSON
        """
        n_iters = queue.qsize()
        pbar = tqdm(total=n_iters)
        pbar.set_description('Requesting SPARQL endpoint for each item')
        
        # instantiating each Worker
        workers = []
        for _ in range(self.n_workers):
            worker = Worker(queue, lambda q: self._query(q, return_type), pbar)
            worker.start()
            workers.append(worker)

        for worker in workers:
            worker.join()
        pbar.close()

        # combining results
        responses = []
        for worker in workers:
            responses.extend(worker.local_results)
        
        return responses
    
    def sequential_queries(self, q, return_type=JSON):
        """
        Sequential query SPARQL endpoint
        :arguments: 
            queue: queue of tuples indicating the item_id and SPARQL query string
        :returns: list containing tuples with item_id and response result in JSON
        """
        pbar = tqdm(total=q.qsize())
        pbar.set_description('Requesting SPARQL endpoint for each item')
        responses = []

        while True:
            try:
                idx, query = q.get(block=False)
                response = self._query(query, return_type)
                responses.append((idx, response))
                pbar.update(n=1)
            except queue.Empty:
                break
            except Exception as e:
                print(f'Exception:')
                print(e)

        return responses

    def _query(self, query, return_type=JSON) -> dict():
        sparql = SPARQLWrapper(self.sparql_endpoint)
        sparql.setTimeout(self.timeout)
        sparql.setQuery(query)
        sparql.setReturnFormat(return_type)

        try:
            return sparql.query().convert()
        except Exception as e:
            raise e
    
    def convert_item_data(self):
        """
        Converts loaded item data to a processed csv (item.csv).
        """
        try:
            print(f'Creating file: {self.item_filename}.')
            df_item = self.load_item_data()
            print(f'{df_item.shape[0]} items with {df_item.shape[1]} Fields')
            print('Fields: ' + ', '.join(self.item_fields))
            df_item.to_csv(self.item_filename, index=False)
        except NotImplementedError:
            print('Override load_item_data() method of your Dataset subclass.')
    
    def convert_user_data(self):
        """
        Converts loaded user data to a processed csv (user.csv). 
        """
        try:
            print(f'Creating file: {self.user_filename}')
            df_user = self.load_user_data()
            print(f'{df_user.shape[0]} users with {df_user.shape[1]} Fields')
            print('Fields: ' + ', '.join(self.user_fields))
            df_user.to_csv(self.user_filename, index=False)
        except NotImplementedError:
            print(f'Override load_user_data() of your Dataset subclass.')

    def convert_rating_data(self):
        """
        Converts loaded rating data to a processed csv (rating.csv)
        """
        try:
            print(f'Creating file: {self.rating_filename}')
            df_rating = self.load_rating_data()
            print(f'{df_rating.shape[0]} ratings with {df_rating.shape[1]} Fields')
            print('Fields: ' + ', '.join(self.rating_fields))
            df_rating.to_csv(self.rating_filename, index=False)
        except NotImplementedError:
            print(f'Override load_rating_data() of your Dataset subclass.')

    def map_URIs(self):
        """
        Maps each item to their corresponding DBpedia's URI.
        """
        try:
            df_item = pd.read_csv(self.item_filename)
            print(f'Mapping each dataset item with DBpedia: {self.map_filename}')
            df_map = self.entity_linking(df_item)
            df_map.to_csv(self.map_filename, index=False)
            
            # print mapping statistics
            n_items = df_map.shape[0]
            n_unmatched = df_map['URI'].isna().sum()
            print(f"{n_unmatched} items weren't matched, corresponding to {n_unmatched/n_items*100:.2f}% of a total of {n_items}.")
            

        except NotImplementedError:
            print('Override entity_linking() method of your Dataset subclass.')

    def enrich_data(self):
        """
        Enrich each mapped item using DBpedia's resources.
        """

        try:
            df_map = pd.read_csv(self.map_filename)
            print(f'Enriching each item with DBpedia resources: {self.enriched_filename}')
            df_enrich = self.enrich(df_map)
            df_enrich.to_csv(self.enriched_filename)            

        except NotImplementedError:
            print('Override entity_linking() method of your Dataset subclass.')
    
    def get_enriching_query(self, URI, metadata):
        relations, patterns = [], []
        for r in metadata['properties']:
            relations.append(f'?{r["relation"]}')
            patterns.append(f'\t\t\t\tOPTIONAL {{ <{URI}> {r["property"]} ?{r["relation"]} }} .')
        
        params = {
            "relations": ' '.join(relations),
            "patterns":  '\n'.join(patterns)
        }
        return self.enrich_query_template.substitute(params)




    


        
        