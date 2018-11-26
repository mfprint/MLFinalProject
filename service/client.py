import json
import numpy as np
import keras
import keras.preprocessing.text as kpt
from keras.preprocessing.text import Tokenizer
from keras.models import model_from_json
import tensorflow as tf

class Client():
    def __init__(self):
        self.tokenizer = Tokenizer(num_words=10000)
        with open('./ai/dictionary.json', 'r') as dictionary_file:
            self.dictionary = json.load(dictionary_file)

        json_file = open('./ai/model.h5', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        self.model = model_from_json(loaded_model_json)
        self.model.load_weights('./ai/weights.h5')
        self.graph = tf.get_default_graph()

    def convert_text_to_index_array(self, text):
        words = kpt.text_to_word_sequence(text)
        wordIndices = []
        for word in words:
            if word in self.dictionary:
                wordIndices.append(self.dictionary[word])
            else:
                print("'%s' not in training corpus; ignoring." %(word))
        return wordIndices

    def predict(self, text):
        testArr = self.convert_text_to_index_array(text)
        input = self.tokenizer.sequences_to_matrix([testArr], mode='binary')

        with self.graph.as_default():
            pred = self.model.predict(input)
        return pred
