import stow
from datetime import datetime
import sys

working_dir = 'H:\Desktop_Files\CSE472\Project\Mltu\mltu'
sys.path.insert(0, working_dir)

from mltu.configs import BaseModelConfigs

class ModelConfigs(BaseModelConfigs):
    def __init__(self):
        super().__init__()
        # self.model_path = stow.join('Models/04_sentence_recognition', datetime.strftime(datetime.now(), "%Y%m%d%H%M"))
        self.model_path = stow.join('Models', 'yaml')
        self.vocab = 'abcdefghijklmnopqrstuvwxyz0123456789,;.!?:\'"/\\|_@#$%^&*~`+-=<>()[]\{\} ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        self.height = 96
        self.width = 1408
        self.max_text_length = 0
        self.batch_size = 32
        self.learning_rate = 0.001
        self.train_epochs = 10
        self.train_workers = 20
        self.dataset_path = 'H:\Desktop_Files\CSE472\Project\Dataset'
        self.working_dir = working_dir

class ModelConfigs2(BaseModelConfigs):
    def __init__(self):
        super().__init__()
        # self.model_path = stow.join('Models/04_sentence_recognition', datetime.strftime(datetime.now(), "%Y%m%d%H%M"))
        self.model_path = stow.join('Models', 'yaml2')
        self.vocab = 'abcdefghijklmnopqrstuvwxyz0123456789,;.!?:\'"/\\|_@#$%^&*~`+-=<>()[]\{\} ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        self.height = 96
        self.width = 1408
        self.max_text_length = 0
        self.batch_size = 32
        self.learning_rate = 0.001
        self.train_epochs = 5
        self.train_workers = 20
        self.dataset_path = 'H:\Desktop_Files\CSE472\Project\Dataset'
        self.working_dir = working_dir