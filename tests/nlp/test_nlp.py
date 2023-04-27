import os
import unittest
from dotenv import load_dotenv

from hugging_py_face.nlp import NLP

load_dotenv()


class TestNLP(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.nlp = NLP(os.environ.get("API_KEY"))

    def test_fill_mask(self):
        text = "The answer to the universe is [MASK]."

        self.assertEqual(
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
        )

    def test_summarization(self):
        text = "The tower is 324 metres (1,063 ft) tall, about the same height as an 81-storey building, and the tallest structure in Paris. Its base is square, measuring 125 metres (410 ft) on each side. During its construction, the Eiffel Tower surpassed the Washington Monument to become the tallest man-made structure in the world, a title it held for 41 years until the Chrysler Building in New York City was finished in 1930. It was the first structure to reach a height of 300 metres. Due to the addition of a broadcasting aerial at the top of the tower in 1957, it is now taller than the Chrysler Building by 5.2 metres (17 ft). Excluding transmitters, the Eiffel Tower is the second tallest free-standing structure in France after the Millau Viaduct."

        self.assertEqual(
            self.nlp.summarization(text),
            [
                {
                    "summary_text": "The tower is 324 metres (1,063 ft) tall, about the same height as an 81-storey building. Its base is square, measuring 125 metres (410 ft) on each side. During its construction, the Eiffel Tower surpassed the Washington Monument to become the tallest man-made structure in the world.",
                },
            ],
        )

    def test_question_answering(self):
        question = "What's my name?"
        context = "My name is Clara and I live in Berkeley"

        self.assertEqual(
            self.nlp.question_answering(question, context),
            {
                "score": 0.7940344214439392,
                "start": 11,
                "end": 16,
                "answer": "Clara"
            }
        )

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

        self.assertEqual(
            self.nlp.table_question_answering(question, table),
            {
                "answer": "AVERAGE > 36542",
                "coordinates": [[0, 1]],
                "cells": ["36542"],
                "aggregator": "AVERAGE",
            },
        )

    def test_sentence_similarity(self):
        source_sentence = "That is a happy person"
        sentences = ["That is a happy dog", "That is a very happy person", "Today is a sunny day"]

        self.assertEqual(
            self.nlp.sentence_similarity(source_sentence, sentences),
            [0.6945773363113403, 0.9429150819778442, 0.2568760812282562],
        )

    def test_text_classification(self):
        text = "I like you. I love you"

        self.assertEqual(
            self.nlp.text_classification(text),
            [
                [
                    {"label": "POSITIVE", "score": 0.9998738765716553},
                    {"label": "NEGATIVE", "score": 0.00012611244164872915},
                ]
            ],
        )

    def test_text_generation(self):
        text = "The answer to the universe is"

        self.assertEqual(
            self.nlp.text_generation(text),
            [
                {'generated_text': "The answer to the universe is in one's own minds and your "
                                'thoughts.\n'
                                '\n'
                                'When the universe is created to bring forth new creation, '
                                'our minds are born anew. The universe is a process '
                                'wherein all things are possible and the universe is one '
                                'created'
                 }
            ]
        )

    def test_zero_shot_classification(self):
        text = "Hi, I recently bought a device from your company but it is not working as advertised and I would like to get reimbursed!"
        candidate_labels = ["refund", "legal", "faq"]

        self.assertEqual(
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
        )

    def test_conversational(self):
        past_user_inputs = ["Which movie is the best ?"]
        generated_responses = ["It's Die Hard for sure."]
        text = "Can you explain why ?"

        self.assertEqual(
            self.nlp.conversational(text, past_user_inputs, generated_responses),
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
                "warnings": ["Setting `pad_token_id` to `eos_token_id`:50256 for open-end generation."],
            },
        )

    def test_feature_extraction(self):
        text = "Transformers is an awesome library!"

        self.assertEqual(type(self.nlp.feature_extraction(text)), list)

