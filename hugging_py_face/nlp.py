import json
import time
import requests
import pandas as pd
from pandas import DataFrame
from typing import Text, List, Dict, Optional, Union

from .base_api import BaseAPI
from .exceptions import HTTPServiceUnavailableException, APICallException, InsufficientParametersException


class NLP(BaseAPI):
    def __init__(self, api_token: Text, api_url: Optional[Text] = None):
        super().__init__(api_token, api_url)

    def _query(self, inputs: Union[Text, List, Dict], parameters: Optional[Dict] = None, options: Optional[Dict] = None, model: Optional[Text] = None, task: Optional[Text] = None) -> Union[Dict, List]:
        if model:
            self._check_model_task_match(model, task)

        api_url = f"{self.api_url}/{model if model is not None else self.config['TASK_MODEL_MAP'][task]}"

        headers = {
            "Authorization": f"Bearer {self.api_token}"
        }

        data = {
            "inputs": inputs
        }

        if parameters is not None:
            data['parameters'] = parameters

        if options is not None:
            data['options'] = options

        retries = 0

        while retries < self.config['MAX_RETRIES']:
            retries += 1

            response = requests.request("POST", api_url, headers=headers, data=json.dumps(data))
            if response.status_code == int(self.config['HTTP_SERVICE_UNAVAILABLE']):
                self.logger.info(f"Status code: {response.status_code}.")
                self.logger.info("Retrying..")
                time.sleep(1)
            elif response.status_code == 200:
                return json.loads(response.content.decode("utf-8"))
            else:
                self.logger.info(f"Status code: {response.status_code}.")
                error_message = self._extract_error_message(response)
                raise APICallException(f"API call failed with the error: {error_message}.")

        self.logger.info(f"Status code: {response.status_code}.")
        self.logger.info("Connection to the server failed after reaching maximum retry attempts.")
        self.logger.debug(f"Response: {json.loads(response.content.decode('utf-8'))}.")
        raise HTTPServiceUnavailableException("The HTTP service is unavailable.")

    def _query_in_df(self, df: DataFrame, column: Text, parameters: Optional[Dict] = None, options: Optional[Dict] = None, model: Optional[Text] = None, task: Optional[Text] = None) -> Union[Dict, List]:
        return self._query(df[column].tolist(), parameters, options, model, task)

    def fill_mask(self, text: Union[Text, List], options: Optional[Dict] = None, model: Optional[Text] = None) -> List:
        """
        Fill in a masked portion(token) of a string or a list of strings.

        :param text: a string or list of strings to be filled. Each input must contain the [MASK] token.
        :param options: a dict of options. For more information, see the `detailed parameters for the fill mask task <https://huggingface.co/docs/api-inference/detailed_parameters#fill-mask-task>`_.
        :param model: the model to use for the fill mask task. If not provided, the recommended model from Hugging Face will be used.
        :return: a list of dicts or a list of lists (of dicts) containing the possible completions and their associated probabilities.
        """
        return self._query(text, options=options, model=model, task='fill-mask')

    def fill_mask_in_df(self, df: DataFrame, column: Text, options: Optional[Dict] = None, model: Optional[Text] = None) -> DataFrame:
        """
        Fill in the masked portion(token) of a column of strings in a DataFrame.

        :param df: a pandas DataFrame containing the strings to be filled.
        :param column: the column containing the strings to be filled.
        :param options: a dict of options. For more information, see the `detailed parameters for the fill mask task <https://huggingface.co/docs/api-inference/detailed_parameters#fill-mask-task>`_.
        :param model: the model to use for the fill mask task. If not provided, the recommended model from Hugging Face will be used.
        :return: a pandas DataFrame with the completions for the masked strings. Each completion added will be the one with the highest probability for that particular masked string. The completions will be added as a new column called 'predictions' to the original DataFrame.
        """
        predictions = self._query_in_df(df, column, options=options, model=model, task='fill-mask')

        if any(isinstance(prediction, list) for prediction in predictions):
            df['predictions'] = [prediction[0]['sequence'] for prediction in predictions]
        else:
            df['predictions'] = [predictions[0]['sequence']]

        return df

    def summarization(self, text: Union[Text, List], parameters: Optional[Dict] = None, options: Optional[Dict] = None, model: Optional[Text] = None) -> Union[Dict, List]:
        """
        Summarize a string or a list of strings.

        :param text: a string or list of strings to be summarized.
        :param parameters: a dict of parameters. For more information, see the `detailed parameters for the summarization task <https://huggingface.co/docs/api-inference/detailed_parameters#summarization-task>`_.
        :param options: a dict of options. For more information, see the `detailed parameters for the summarization task <https://huggingface.co/docs/api-inference/detailed_parameters#summarization-task>`_.
        :param model: the model to use for the summarization task. If not provided, the recommended model from Hugging Face will be used.
        :return: a dict or a list of dicts of the summarized string(s).
        """
        return self._query(text, parameters=parameters, options=options, model=model, task='summarization')

    def summarization_in_df(self, df: DataFrame, column: Text, parameters: Optional[Dict] = None, options: Optional[Dict] = None, model: Optional[Text] = None) -> DataFrame:
        """
        Summarize a column of strings in a DataFrame.

        :param df: a pandas DataFrame containing the strings to be summarized.
        :param column: the column containing the strings to be summarized.
        :param parameters: a dict of parameters. For more information, see the `detailed parameters for the summarization task <https://huggingface.co/docs/api-inference/detailed_parameters#summarization-task>`_.
        :param options: a dict of options. For more information, see the `detailed parameters for the summarization task <https://huggingface.co/docs/api-inference/detailed_parameters#summarization-task>`_.
        :param model: the model to use for the summarization task. If not provided, the recommended model from Hugging Face will be used.
        :return: a pandas DataFrame with the summarizations for the strings. The summarizations will be added as a new column called 'predictions' to the original DataFrame.
        """
        predictions = self._query_in_df(df, column, parameters=parameters, options=options, model=model, task='summarization')
        df['predictions'] = [prediction['summary_text'] for prediction in predictions]
        return df

    def question_answering(self, question: Text, context: Text, model: Optional[Text] = None) -> Dict:
        """
        Answer a question using the provided context.

        :param question: a string of the question to be answered.
        :param context: a string of context. This field is required for the question answering task and cannot be left empty.
        :param model: the model to use for the question answering task. If not provided, the recommended model from Hugging Face will be used.
        :return: a dict of the answer.
        """
        return self._query(
            {
                "question": question,
                "context": context
            },
            model=model,
            task='question-answering'
        )

    def question_answering_in_df(self, df: DataFrame, question_column: Text, context_column: Text, model: Optional[Text] = None) -> DataFrame:
        """
        Generate answers for a column of questions based on a provided column of context.

        :param df: a pandas DataFrame containing the questions to be answered along with the relevant context.
        :param question_column: the column containing the questions to be answered.
        :param context_column: the column containing the relevant context for each question.
        :param model: the model to use for the question answering task. If not provided, the recommended model from Hugging Face will be used.
        :return: a pandas DataFrame with the answers for the questions. The answers will be added as a new column called 'predictions' to the original DataFrame.
        """
        answers = []
        for index, row in df.iterrows():
            answer = self._query(
                {
                    "question": row[question_column],
                    "context": row[context_column]
                },
                model=model,
                task='question-answering'
            )
            answers.append(answer['answer'])

        df['predictions'] = answers
        return df

    def table_question_answering(self, question: Union[Text, List], table: Dict[Text, List], options: Optional[Dict] = None, model: Optional[Text] = None) -> List:
        """

        :param question: a string or a list of strings of the question(s) to be answered.
        :param table: a dict of lists representing a table of data.
        :param options: a dict of options. For more information, see the `detailed parameters for the table question answering task <https://huggingface.co/docs/api-inference/detailed_parameters#table-question-answering-task>`_.
        :param model: the model to use for the table question answering task. If not provided, the recommended model from Hugging Face will be used.
        :return: a dict or a list of dicts of the answers.
        """
        return self._query(
            {
                "query": question,
                "table": table
            },
            options=options,
            model=model,
            task='table-question-answering'
        )

    def table_question_answering_task_in_df(self, df: DataFrame, question: Union[Text, List], options: Optional[Dict] = None, model: Optional[Text] = None) -> DataFrame:
        answers = self._query(
            {
                "query": question,
                "table": df.to_dict('list')
            },
            options=options,
            model=model,
            task='table-question-answering'
        )

        return pd.DataFrame({
            "question": question,
            "predictions": [answer['answer'] for answer in answers]
        })

    def sentence_similarity(self, source_sentence: Text, sentences: List, options: Optional[Dict] = None, model: Optional[Text] = None) -> List:
        """
        Calculate the semantic similarity between one text and a list of other sentences by comparing their embeddings.

        :param source_sentence: the string that you wish to compare the other strings with.
        :param sentences: a list of strings which will be compared against the source_sentence.
        :param options: a dict of options. For more information, see the `detailed parameters for the sentence similarity task <https://huggingface.co/docs/api-inference/detailed_parameters#sentence-similarity-task>`_.
        :param model: the model to use for the sentence similarity task. If not provided, the recommended model from Hugging Face will be used.
        :return: a list of similarity scores.
        """
        return self._query(
            {
                "source_sentence": source_sentence,
                "sentences": sentences
            },
            options=options,
            model=model,
            task='sentence-similarity'
        )

    def sentence_similarity_in_df(self, df: DataFrame, source_sentence_column: Text, sentence_column: Text, options: Optional[Dict] = None, model: Optional[Text] = None) -> DataFrame:
        """
        Calculate the semantic similarity between sentences in two columns by comparing their embeddings.

        :param df: a pandas DataFrame containing the source sentences and the sentences to be compared against.
        :param source_sentence_column: the column containing the strings that you wish to compare the other strings with.
        :param sentence_column: the column containing the strings which will be compared against the source_sentence.
        :param options: a dict of options. For more information, see the `detailed parameters for the sentence similarity task <https://huggingface.co/docs/api-inference/detailed_parameters#sentence-similarity-task>`_.
        :param model: the model to use for the sentence similarity task. If not provided, the recommended model from Hugging Face will be used.
        :return: a pandas DataFrame with the similarity scores for the sentences. The scores will be added as a new column called 'predictions' to the original DataFrame.
        """
        scores = []
        for index, row in df.iterrows():
            score = self._query(
                {
                    "source_sentence": row[source_sentence_column],
                    "sentences": [row[sentence_column]]
                },
                options=options,
                model=model,
                task='sentence-similarity'
            )
            scores.append(score[0])

        df['predictions'] = scores
        return df

    def text_classification(self, text: Union[Text, List], options: Optional[Dict] = None, model: Optional[Text] = None) -> Union[Dict, List]:
        """
        Analyze the sentiment of a string or a list of strings.

        :param text: a string or list of strings to be analyzed.
        :param options: a dict of options. For more information, see the `detailed parameters for the summarization task <https://huggingface.co/docs/api-inference/detailed_parameters#text-classification-task>`_.
        :param model: the model to use for the text classification task. If not provided, the recommended model from Hugging Face will be used.
        :return: a dict or a list of dicts indicating the possible sentiments of the string(s) and their associated probabilities.
        """
        return self._query(text, options=options, model=model, task='text-classification')

    def text_classification_in_df(self, df: DataFrame, column: Text, options: Optional[Dict] = None, model: Optional[Text] = None) -> DataFrame:
        """
        Analyze the sentiment of a column of strings in a DataFrame.

        :param df: a pandas DataFrame containing the strings to be analyzed.
        :param column: the column containing the strings to be analyzed.
        :param options: a dict of options. For more information, see the `detailed parameters for the summarization task <https://huggingface.co/docs/api-inference/detailed_parameters#text-classification-task>`_.
        :param model: the model to use for the text classification task. If not provided, the recommended model from Hugging Face will be used.
        :return: a pandas DataFrame with the sentiment of the strings. Each sentiment added will be the one with the highest probability for that particular string. The sentiment will be added as a new column called 'predictions' to the original DataFrame.
        """
        predictions = self._query_in_df(df, column, options=options, model=model, task='text-classification')
        df['predictions'] = [prediction[0]['label'] for prediction in predictions]
        return df

    def text_generation(self, text: Union[Text, List], parameters: Optional[Dict] = None, options: Optional[Dict] = None, model: Optional[Text] = None) -> Union[Dict, List]:
        """
        Continue text from a prompt.

        :param text: a string to be generated from.
        :param parameters: a dict of parameters. For more information, see the `detailed parameters for the text generation task <https://huggingface.co/docs/api-inference/detailed_parameters#text-generation-task>`_.
        :param options: a dict of options. For more information, see the `detailed parameters for the text generation task <https://huggingface.co/docs/api-inference/detailed_parameters#text-generation-task>`_.
        :param model: the model to use for the text generation task. If not provided, the recommended model from Hugging Face will be used.
        :return: a dict or a list of dicts containing the generated text.
        """
        return self._query(text, parameters=parameters, options=options, model=model, task='text-generation')

    def text_generation_in_df(self, df: DataFrame, column: Text, parameters: Optional[Dict] = None, options: Optional[Dict] = None, model: Optional[Text] = None) -> DataFrame:
        """
        Continue text from a prompt in the column of a DataFrame.

        :param df: a pandas DataFrame containing the strings to be generated from.
        :param column: the column containing the strings to be generated from.
        :param parameters: a dict of parameters. For more information, see the `detailed parameters for the text generation task <https://huggingface.co/docs/api-inference/detailed_parameters#text-generation-task>`_.
        :param options: a dict of options. For more information, see the `detailed parameters for the text generation task <https://huggingface.co/docs/api-inference/detailed_parameters#text-generation-task>`_.
        :param model: the model to use for the text generation task. If not provided, the recommended model from Hugging Face will be used.
        :return: a pandas DataFrame with the generated text. The generated text will be added as a new column called 'predictions' to the original DataFrame.
        """
        predictions = self._query_in_df(df, column, parameters=parameters, options=options, model=model, task='text-generation')
        df['predictions'] = [prediction[0]['generated_text'] for prediction in predictions]
        return df

    def zero_shot_classification(self, text: Union[Text, List], candidate_labels: List, parameters: Optional[Dict] = {}, options: Optional[Dict] = None, model: Optional[Text] = None) -> Union[Dict, List]:
        """
        Classify a sentence/paragraph to one of the candidate labels provided.

        :param text: a string or list of strings to be classified.
        :param candidate_labels: a list of strings that are potential classes for inputs.
        :param parameters: a dict of parameters excluding candidate_labels which is passed in as a separate argument. For more information, see the `detailed parameters for the zero shot classification task <https://huggingface.co/docs/api-inference/detailed_parameters#zeroshot-classification-task>`_.
        :param options: a dict of options. For more information, see the `detailed parameters for the zero shot classification task <https://huggingface.co/docs/api-inference/detailed_parameters#zeroshot-classification-task>`_.
        :param model: the model to use for the zero shot classification task. If not provided, the recommended model from Hugging Face will be used.
        :return: a dict or a list of dicts containing the labels and the corresponding the probability of each label.
        """
        parameters['candidate_labels'] = candidate_labels

        return self._query(
            text,
            parameters=parameters,
            options=options,
            model=model,
            task='zero-shot-classification'
        )

    def zero_shot_classification_in_df(self, df: DataFrame, column: Text, candidate_labels: List, parameters: Optional[Dict] = {}, options: Optional[Dict] = None, model: Optional[Text] = None):
        """

        :param df: a pandas DataFrame containing the strings to be classified.
        :param column: the column containing the strings to be classified.
        :param candidate_labels: a list of strings that are potential classes for inputs.
        :param parameters: a dict of parameters excluding candidate_labels which is passed in as a separate argument. For more information, see the `detailed parameters for the zero shot classification task <https://huggingface.co/docs/api-inference/detailed_parameters#zeroshot-classification-task>`_.
        :param options: a dict of options. For more information, see the `detailed parameters for the zero shot classification task <https://huggingface.co/docs/api-inference/detailed_parameters#zeroshot-classification-task>`_.
        :param model: the model to use for the zero shot classification task. If not provided, the recommended model from Hugging Face will be used.
        :return: a pandas DataFrame with the classifications. The classifications will be added as a new column called 'predictions' to the original DataFrame.
        """
        parameters['candidate_labels'] = candidate_labels
        predictions = self._query_in_df(df, column, parameters=parameters, options=options, model=model, task='zero-shot-classification')
        df['predictions'] = [prediction['labels'][0] for prediction in predictions]
        return df

    def conversational(self, text: Union[Text, List], past_user_inputs: Optional[List] = None, generated_responses: Optional[List] = None, parameters: Optional[Dict] = None, options: Optional[Dict] = None, model: Optional[Text] = None) -> Union[Dict, List]:
        """
        Corresponds to any chatbot like structure: pass in some text along with the past_user_inputs and generated_responses to receive a response.

        :param text: a string or list of strings representing the last input(s) from the user in the conversation.
        :param past_user_inputs: a list of strings corresponding to the earlier replies from the user. Should be of the same length of generated_responses. Each response from the bot will contain past_user_inputs previously passed into the model.
        :param generated_responses: a list of strings corresponding to the earlier replies from the model. Each response from the bot will contain generated_responses from earlier replies from the model.
        :param parameters: a dict of parameters. For more information, see the `detailed parameters for the conversational task <https://huggingface.co/docs/api-inference/detailed_parameters#conversational-task>`_.
        :param options: a dict of options. For more information, see the `detailed parameters for the conversational task <https://huggingface.co/docs/api-inference/detailed_parameters#conversational-task>`_.
        :param model: the model to use for the conversational task. If not provided, the recommended model from Hugging Face will be used.
        :return: a dict or a list of dicts containing the response(s) from the bot.
        """
        inputs = {
            'text': text
        }

        if past_user_inputs is not None:
            inputs['past_user_inputs'] = past_user_inputs

        if generated_responses is not None:
            inputs['generated_responses'] = generated_responses

        return self._query(
            inputs,
            parameters=parameters,
            options=options,
            model=model,
            task='conversational'
        )

    def feature_extraction(self, text: Union[Text, List], options: Optional[Dict] = None, model: Optional[Text] = None) -> Union[Dict, List]:
        """
        Reads some text and outputs raw float values, that are usually consumed as part of a semantic database/semantic search.

        :param text: a string or a list of strings to get the features from.
        :param options: a dict of options. For more information, see the `detailed parameters for the feature extraction task <https://huggingface.co/docs/api-inference/detailed_parameters#feature-extraction-task>`_.
        :param model: the model to use for the feature extraction task. If not provided, the recommended model from Hugging Face will be used.
        :return: a list of dicts or a list of lists (of dicts) containing the representation of the features of the input(s).
        """
        return self._query(text, options=options, model=model, task='feature-extraction')

    def translation(self, text: Union[Text, List], lang_input: Text = None, lang_output: Text = None, options: Optional[Dict] = None, model: Optional[Text] = None) -> Union[Dict, List]:
        """
        Translates text from one language to another.

        :param text: a string or a list of strings to translate.
        :param lang_input: the short code of the language of the input text. This parameter is mandatory if the model is not provided.
        :param lang_output: the short code of the language to translate the input text to. This parameter is mandatory if the model is not provided.
        :param options: a dict of options. For more information, see the `detailed parameters for the translation task <https://huggingface.co/docs/api-inference/detailed_parameters#translation-task>`_.
        :param model: the model to use for the translation task. If not provided, the recommended model from Hugging Face will be used.
        :return: a dict or a list of dicts containing the translated text.
        """
        if model is None:
            if lang_input is None or lang_output is None:
                InsufficientParametersException("lang_input and lang_output are required if model is not provided.")
            model = f"{self.config['TASK_MODEL_MAP']['translation']}{lang_input}-{lang_output}"
            return self._query(text, options=options, model=model, task='translation')
        else:
            return self._query(text, options=options, model=model, task='translation')

    def translation_in_df(self, df: DataFrame, column: Text, lang_input: Text = None, lang_output: Text = None, options: Optional[Dict] = None, model: Optional[Text] = None) -> DataFrame:
        """
        Translates text from one language to another.

        :param df: a pandas DataFrame containing the strings to be translated.
        :param column: the column containing the strings to be translated.
        :param lang_input: the short code of the language of the input text. This parameter is mandatory if the model is not provided.
        :param lang_output: the short code of the language to translate the input text to. This parameter is mandatory if the model is not provided.
        :param options: a dict of options. For more information, see the `detailed parameters for the translation task <https://huggingface.co/docs/api-inference/detailed_parameters#translation-task>`_.
        :param model: the model to use for the translation task. If not provided, the recommended model from Hugging Face will be used.
        :return: a pandas DataFrame with the translations. The translations will be added as a new column called 'predictions' to the original DataFrame.
        """
        if model is None:
            if lang_input is None or lang_output is None:
                InsufficientParametersException("lang_input and lang_output are required if model is not provided.")
            model = f"{self.config['TASK_MODEL_MAP']['translation']}{lang_input}-{lang_output}"
            predictions = self._query_in_df(df, column, options=options, model=model, task='translation')
        else:
            predictions = self._query_in_df(df, column, options=options, model=model, task='translation')

        df['predictions'] = [prediction['translation_text'] for prediction in predictions]
        return df