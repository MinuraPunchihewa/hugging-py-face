import os
import unittest
from dotenv import load_dotenv

from hugging_py_face.nlp import NLP
from hugging_py_face.exceptions import HTTPServiceUnavailableException

load_dotenv()


class TestNLP(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.nlp = NLP(os.environ.get("API_KEY"))

    def test_fill_mask(self):
        text = "The answer to the universe is [MASK]."

        def assert_almost_equal_list(expected, actual, places=7):
            self.assertEqual(len(expected), len(actual))
            for exp, act in zip(expected, actual):
                self.assertEqual(exp['sequence'], act['sequence'])
                self.assertEqual(exp['token'], act['token'])
                self.assertEqual(exp['token_str'], act['token_str'])
                self.assertAlmostEqual(exp['score'], act['score'], places=places)

        try:
            assert_almost_equal_list(
                self.nlp.fill_mask(text),
                [
                    {
                        "sequence": "the answer to the universe is no.",
                        "score": 0.16963981091976166,
                        "token": 2053,
                        "token_str": "no",
                    },
                    {
                        "sequence": "the answer to the universe is nothing.",
                        "score": 0.07344783842563629,
                        "token": 2498,
                        "token_str": "nothing",
                    },
                    {
                        "sequence": "the answer to the universe is yes.",
                        "score": 0.05803249776363373,
                        "token": 2748,
                        "token_str": "yes",
                    },
                    {
                        "sequence": "the answer to the universe is unknown.",
                        "score": 0.043957870453596115,
                        "token": 4242,
                        "token_str": "unknown",
                    },
                    {
                        "sequence": "the answer to the universe is simple.",
                        "score": 0.040157340466976166,
                        "token": 3722,
                        "token_str": "simple",
                    },
                ],
                4
            )
        except HTTPServiceUnavailableException:
            self.assertRaises(HTTPServiceUnavailableException, lambda: self.nlp.fill_mask(text))

    def test_summarization(self):
        text = "The tower is 324 metres (1,063 ft) tall, about the same height as an 81-storey building, and the tallest structure in Paris. Its base is square, measuring 125 metres (410 ft) on each side. During its construction, the Eiffel Tower surpassed the Washington Monument to become the tallest man-made structure in the world, a title it held for 41 years until the Chrysler Building in New York City was finished in 1930. It was the first structure to reach a height of 300 metres. Due to the addition of a broadcasting aerial at the top of the tower in 1957, it is now taller than the Chrysler Building by 5.2 metres (17 ft). Excluding transmitters, the Eiffel Tower is the second tallest free-standing structure in France after the Millau Viaduct."

        def check_words_in_string(word_list, string):
            for word in word_list:
                if word in string:
                    return True
            return False

        try:
            self.assertTrue(
                check_words_in_string(
                    ["Eiffel Tower", "324", "81-storey"],
                    self.nlp.summarization(text)[0]['summary_text']
                )
            )
        except HTTPServiceUnavailableException:
            pass

    def test_question_answering(self):
        question = "What's my name?"
        context = "My name is Clara and I live in Berkeley"

        def assert_almost_equal_dict(expected, actual, places=4):
            self.assertEqual(expected['start'], actual['start'])
            self.assertEqual(expected['end'], actual['end'])
            self.assertEqual(expected['answer'], actual['answer'])
            self.assertAlmostEqual(expected['score'], actual['score'], places=places)

        try:
            assert_almost_equal_dict(
                self.nlp.question_answering(question, context),
                {
                    "score": 0.7940344214439392,
                    "start": 11,
                    "end": 16,
                    "answer": "Clara"
                }
            )
        except HTTPServiceUnavailableException:
            pass

    def test_table_question_answering(self):
        question = "How many stars does the transformers repository have?"
        table = {
            "Repository": ["Transformers", "Datasets", "Tokenizers"],
            "Stars": ["36542", "4512", "3934"],
            "Contributors": ["651", "77", "34"],
            "Programming language": [
                "Python",
                "Python",
                "Rust, Python and NodeJS",
            ],
        }

        try:
            self.assertEqual(
                self.nlp.table_question_answering(question, table),
                {
                    "answer": "AVERAGE > 36542",
                    "coordinates": [[0, 1]],
                    "cells": ["36542"],
                    "aggregator": "AVERAGE",
                },
            )
        except HTTPServiceUnavailableException:
            pass

    def test_sentence_similarity(self):
        source_sentence = "That is a happy person"
        sentences = ["That is a happy dog", "That is a very happy person", "Today is a sunny day"]

        def assert_almost_equal_list(expected, actual, decimal_places=3):
            self.assertEqual(len(expected), len(actual), "List lengths are different.")

            for exp, act in zip(expected, actual):
                exp_str = "{:.{}f}".format(exp, decimal_places)
                act_str = "{:.{}f}".format(act, decimal_places)
                self.assertEqual(exp_str, act_str, "Values are not approximately equal.")

        try:
            assert_almost_equal_list(
                self.nlp.sentence_similarity(source_sentence, sentences),
                [0.6945773363113403, 0.9429150819778442, 0.2568760812282562],
            )
        except HTTPServiceUnavailableException:
            pass

    def test_text_classification(self):
        text = "I like you. I love you"

        def assert_almost_equal_list(expected, actual, places=7):
            self.assertEqual(len(expected), len(actual))
            for exp, act in zip(expected, actual):
                self.assertEqual(exp['label'], act['label']), "Label values are not equal."
                self.assertAlmostEqual(exp['score'], act['score'],
                                       places=places), "Score values are not approximately equal."

        try:
            assert_almost_equal_list(
                self.nlp.text_classification(text)[0],
                [
                    {"label": "POSITIVE", "score": 0.9998738765716553},
                    {"label": "NEGATIVE", "score": 0.00012611244164872915},
                ],
            )
        except HTTPServiceUnavailableException:
            pass

    def test_text_generation(self):
        text = "The answer to the universe is"

        try:
            prediction = self.nlp.text_generation(text)
            self.assertTrue(prediction[0]['generated_text'].startswith(text))
        except HTTPServiceUnavailableException:
            pass

    def test_zero_shot_classification(self):
        text = "Hi, I recently bought a device from your company but it is not working as advertised and I would like to get reimbursed!"
        candidate_labels = ["refund", "legal", "faq"]

        def assert_almost_equal_dict(expected, actual, places=7):
            self.assertEqual(expected['sequence'], actual['sequence'])

            self.assertEqual(len(expected['labels']), len(actual['labels']))
            self.assertEqual(expected['labels'], actual['labels'])

            self.assertEqual(len(expected['scores']), len(actual['scores']))
            self.assertEqual(len(expected), len(actual))
            for exp, act in zip(expected['scores'], actual['scores']):
                self.assertAlmostEqual(exp, act, places=places), "Score values are not approximately equal."

        try:
            assert_almost_equal_dict(
                self.nlp.zero_shot_classification(text, candidate_labels),
                {
                    "sequence": "Hi, I recently bought a device from your company but it is not working as advertised and I would like to get reimbursed!",
                    "labels": ["refund", "faq", "legal"],
                    "scores": [
                        # 88% refund
                        0.8777875304222107,
                        0.10522652417421341,
                        0.01698593609035015,
                    ],
                },
                4
            )
        except HTTPServiceUnavailableException:
            pass

    def test_conversational(self):
        past_user_inputs = ["Which movie is the best ?"]
        generated_responses = ["It's Die Hard for sure."]
        text = "Can you explain why ?"

        try:
            actual_result = self.nlp.conversational(text, past_user_inputs, generated_responses)
            del actual_result["warnings"]
            self.assertEqual(
                actual_result,
                {
                    "generated_text": "It's the best movie ever.",
                    "conversation": {
                        "past_user_inputs": [
                            "Which movie is the best ?",
                            "Can you explain why ?",
                        ],
                        "generated_responses": [
                            "It's Die Hard for sure.",
                            "It's the best movie ever.",
                        ],
                    },
                },
            )
        except HTTPServiceUnavailableException:
            pass

    def test_feature_extraction(self):
        text = "Transformers is an awesome library!"

        try:
            self.assertEqual(type(self.nlp.feature_extraction(text)), list)
        except HTTPServiceUnavailableException:
            pass

