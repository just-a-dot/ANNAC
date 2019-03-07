from abc import ABC, abstractmethod

class ModelTemplate(ABC):

    @abstractmethod
    def get_epochs(self):
        pass

    @abstractmethod
    def get_batch_size(self):
        pass

    @abstractmethod
    def get_optimizer(self):
        pass

    @abstractmethod
    def get_loss_function(self):
        pass

    @abstractmethod
    def get_autoencoder(self):
        pass

    @abstractmethod
    def get_encoder(self):
        pass

    @abstractmethod
    def get_decoder(self):
        pass