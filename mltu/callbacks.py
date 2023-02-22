import os
# from tensorflow.keras.callbacks import Callback
from keras.callbacks import Callback

import logging

class Model2onnx(Callback):
    """Converts the model to onnx format after training is finished.
    
    Args:
        saved_model_path (str): Path to the saved .h5 model.
    """
    try:
        import tf2onnx
    except ImportError:
        raise ImportError("tf2onnx not installed, skipping model export to onnx")

    def __init__(self, saved_model_path: str) -> None:
        super().__init__()
        self.saved_model_path = saved_model_path

    def on_train_end(self, logs=None):
        self.model.load_weights(self.saved_model_path)
        self.tf2onnx.convert.from_keras(self.model, output_path=self.saved_model_path.replace(".h5", ".onnx"), )

class TrainLogger(Callback):
    """Logs training metrics to a file.
    
    Args:
        log_path (str): Path to the directory where the log file will be saved.
        log_file (str, optional): Name of the log file. Defaults to 'logs.log'.
        logLevel (int, optional): Logging level. Defaults to logging.INFO.
    """
    def __init__(self, log_path: str, log_file: str='logs.log', logLevel=logging.INFO) -> None:
        super().__init__()
        self.log_path = log_path
        self.log_file = log_file

        if not os.path.exists(log_path):
            os.mkdir(log_path)

        self.logger = logging.getLogger()
        self.logger.setLevel(logLevel)

        self.formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        self.file_handler = logging.FileHandler(os.path.join(self.log_path, self.log_file))
        self.file_handler.setLevel(logLevel)
        self.file_handler.setFormatter(self.formatter)

        self.logger.addHandler(self.file_handler)

    def on_epoch_end(self, epoch: int, logs: dict=None):
        epoch_message = f"Epoch {epoch}; "
        logs_message = "; ".join([f"{key}: {value}" for key, value in logs.items()])
        self.logger.info(epoch_message + logs_message)