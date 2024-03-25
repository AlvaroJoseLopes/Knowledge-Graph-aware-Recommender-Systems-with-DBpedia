import importlib

from framework.dataloader.graph.graph import Graph
from framework.dataloader.graph.node import UserNode, ItemNode
from ...recommender import Recommender

from ...model2class import model2class

class GMatching(Recommender):
    def __init__(
        self,
        config: dict,
        embedding_model_name: str,
        embedding_config: dict,
        embedding_kwargs: dict,
    ):
        super().__init__(config)
        self.embedding_model_name = embedding_model_name
        self.embedding_kwargs = embedding_kwargs
        self.embedding_config = embedding_config

    def name(self):
        text = "GMatching +"
        text += f";embedding_model={self.embedding_model}; embedding_kwargs={self.embedding_kwargs}"
        return text
    
    def train(self, G_train, ratings_train):
        module_name = f'framework.recommender.models.{model2class[self.embedding_model_name]["submodule"]}'
        class_name = model2class[self.embedding_model_name]['class']

        self._embedding_model = getattr(importlib.import_module(module_name), class_name)
        self._embedding_model = self._embedding_model(self.embedding_config, **self.embedding_kwargs)

        self._embedding_model.train(G_train, ratings_train)

        self.fit()

    def fit(self):
        pass