# Copyright 2019 kubeflow.org.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import kfserving
import joblib
import sklearn
import numpy as np
import pandas as pd
import dill
import os
from typing import List, Dict

MODEL_BASENAME = "model"
MODEL_EXTENSIONS = [".joblib", ".pkl", ".pickle"]


class SKLearnModel(kfserving.KFModel): #pylint:disable=c-extension-no-member
    def __init__(self, name: str, model_dir: str):
        super().__init__(name)
        self.name = name
        self.model_dir = model_dir
        self.ready = False

    def load(self):
        model_path = kfserving.Storage.download(self.model_dir)
        paths = [os.path.join(model_path, MODEL_BASENAME + model_extension)
                 for model_extension in MODEL_EXTENSIONS]
        model_file = next(path for path in paths if os.path.exists(path))
        with open(model_file, 'rb') as f:
            try:
                self._model = dill.load(f) #pylint:disable=attribute-defined-outside-init
            except Exception:
                self._model = joblib.load(f) #pylint:disable=attribute-defined-outside-init
        self.ready = True

    def predict(self, request: Dict) -> Dict:
        instances = request["instances"]
        inputs = np.array(instances)
        proba = request.get("probabilities", False)
        if proba:
            result = self._model.predict_proba(inputs).tolist()
        else:
            result = self._model.predict(inputs).tolist()
        return { "predictions" : result }
