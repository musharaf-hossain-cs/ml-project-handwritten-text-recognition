import cv2
import typing
import numpy as np
import sys

sys.path.insert(0, '../..')
sys.path.insert(0, '..')
sys.path.insert(0, 'H:/Desktop_Files/CSE472/Project/Mltu/mltu')
sys.path.insert(0, 'H:/Desktop_Files/CSE472/Project/Mltu/mltu/Tutorials/04_sentence_recognition')


import sys
sys.path.insert(0, '../..')
sys.path.insert(0, '..')
sys.path.insert(0, 'E:/Backup/mltu-practice')
sys.path.insert(0, 'E:/Backup/mltu-practice/Tutorials/04_sentence_recognition')

from mltu.inferenceModel import OnnxInferenceModel
from mltu.utils.text_utils import ctc_decoder, get_cer, get_wer




class ImageToWordModel(OnnxInferenceModel):
    def __init__(self, char_list: typing.Union[str, list], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.char_list = char_list

    def predict(self, image: np.ndarray):
        image = cv2.resize(image, self.input_shape[:2][::-1])

        image_pred = np.expand_dims(image, axis=0).astype(np.float32)

        preds = self.model.run(None, {self.input_name: image_pred})[0]

        text = ctc_decoder(preds, self.char_list)[0]

        return text

if __name__ == "__main__":
    import pandas as pd
    from tqdm import tqdm
    from mltu.configs import BaseModelConfigs

    configs = BaseModelConfigs.load("Models/yaml/configs.yaml")

    model = ImageToWordModel(model_path=configs.model_path, char_list=configs.vocab)

    df = pd.read_csv("Models/yaml/val.csv").values.tolist()

    accum_cer, accum_wer = [], []
    for image_path, label in tqdm(df):
        image = cv2.imread(image_path)

        prediction_text = model.predict(image)

        cer = get_cer(prediction_text, label)
        wer = get_wer(prediction_text, label)
        print(f"Image: {image_path}; Label: ({label}); Prediction: ({prediction_text}); CER: {cer}; WER: {wer}")

        accum_cer.append(cer)
        accum_wer.append(wer)

    print(f"Average CER: {np.average(accum_cer)}, Average WER: {np.average(accum_wer)}")