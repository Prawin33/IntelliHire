# for language model
import torch
from summarizer import Summarizer
from transformers import BertTokenizer, BertForSequenceClassification, BertModel
from transformers import BartForConditionalGeneration, BartTokenizer

import spacy

import numpy as np

# Build the AI
class NLP_Block():
    def __init__(self):
        # Load the BERT model and tokenizer for similarity analysis
        self.bert_model_name = 'bert-base-uncased'
        self.bert_tokenizer = BertTokenizer.from_pretrained(self.bert_model_name)
        self.bert_model_for_sequence_classification = BertForSequenceClassification.from_pretrained(self.bert_model_name)
        self.bert_model = BertModel.from_pretrained(self.bert_model_name)
        # print(f"BERT Model Parameters: {self.bert_model}")
        self.bert_model.eval()

        # # Load the summarizer model
        # self.summarizer = Summarizer()
        # self.title = 'interview_q_and_a'

        # Load the BART model and tokenizer for text summarization
        self.bart_model_name = 'facebook/bart-large-cnn'
        self.bart_tokenizer = BartTokenizer.from_pretrained(self.bart_model_name)
        self.bart_model = BartForConditionalGeneration.from_pretrained(self.bart_model_name)

        # Load the model for spacy
        self.spacy_model = spacy.load("en_core_web_lg")

    # # Function to summarize text using summarizer library from PyPi
    # def summarize_text(self, text):
    #     summary = self.summarizer.get_summary(text=text, title=self.title)
    #     return summary

    # Function to summarize text using the BART model
    def summarize_text(self, text):
        inputs = self.bart_tokenizer([text], max_length=1024, return_tensors='pt', truncation=True)
        summary_ids = self.bart_model.generate(inputs['input_ids'], num_beams=4, min_length=30, max_length=200,
                                     early_stopping=True)
        summary = self.bart_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return summary

    # Function to analyze text similarity using BERT with sequence classifier
    def analyze_similarity_using_bert_with_sequence_classifier(self, input_text, fixed_answers):
        input_tokens = self.bert_tokenizer.encode(input_text, add_special_tokens=True, return_tensors="pt")
        # input_tokens = input_tokens.to(torch.device("cpu"))
        print(f"size of input_tokens: {input_tokens.shape}")
        similarities = []

        with torch.no_grad():
            # Move the input tokens to the desired device
            input_tokens = input_tokens.to(torch.device("cpu"))
            print(f"size of input_tokens: {input_tokens.shape}")
            max_seq_length = input_tokens.size(1)  # Get the maximum sequence length of input_tokens

            for answer in fixed_answers:
                # answer_tokens = self.bert_tokenizer.encode(answer, add_special_tokens=True, return_tensors="pt")
                # answer_tokens = answer_tokens.to(torch.device("cpu"))
                # print(f"size of answer_tokens: {answer_tokens}")

                # Encode the current answer
                answer_tokens = self.bert_tokenizer.encode(answer, add_special_tokens=True, return_tensors="pt")
                # Pad answer_tokens if its length is less than max_seq_length
                if answer_tokens.size(1) < max_seq_length:
                    pad_length = max_seq_length - answer_tokens.size(1)
                    padding = torch.zeros((1, pad_length), dtype=torch.long)
                    answer_tokens = torch.cat([answer_tokens, padding], dim=1)
                # Move the answer tokens to the desired device
                answer_tokens = answer_tokens.to(torch.device("cpu"))
                print(f"size of answer_tokens: {answer_tokens.shape}")

                outputs = self.bert_model_for_sequence_classification(input_ids=input_tokens, labels=answer_tokens)
                logits = outputs.logits

                probabilities = torch.softmax(logits, dim=1).tolist()[0]
                similarity_score = probabilities[1]  # Probability of being a similar sentence
                similarities.append(similarity_score)

        return similarities

    # Function to analyze text similarity using BERT
    def analyze_similarity_using_bert(self, input_text, fixed_answers):
        similarities = []

        input_tokens = self.bert_tokenizer.encode(input_text, add_special_tokens=True, return_tensors="pt")
        # input_tokens = input_tokens.to(torch.device("cpu"))
        print(f"size of input_tokens: {input_tokens.shape}")

        encoding1 = self.bert_model(input_tokens)
        print(f"outputs for input_tokens: {encoding1}")

        # # Tokenize and encode the texts
        # # text1 = "This is the first text."
        # # text2 = "This is the second text."
        # encoding1 = self.bert_model.(input_text, max_length=512)
        #
        for answer in fixed_answers:
            answer_tokens = self.bert_tokenizer.encode(answer, add_special_tokens=True, return_tensors="pt")
            print(f"size of answer_tokens: {answer_tokens.shape}")

            encoding2 = self.bert_model(answer_tokens)
            print(f"outputs for answer_tokens: {encoding2}")

            # Calculate the cosine similarity between the embeddings
            similarity = np.dot(encoding1, encoding2) / (np.linalg.norm(encoding1) * np.linalg.norm(encoding2))
            print(similarity)
        #
        #     similarities.append(similarity)

        return similarities

    # Function to analyze text similarity using BERT
    def analyze_similarity_using_spacy(self, input_text, fixed_answers):
        similarities = []

        encoding1 = self.spacy_model(input_text)
        print(f"encoding1: {encoding1}")

        # # Tokenize and encode the texts
        for answer in fixed_answers:
            encoding2 = self.spacy_model(answer)
            print(f"outputs for answer_tokens: {encoding2}")

            # Calculate the cosine similarity between the embeddings
            similarity = encoding1.similarity(encoding2)
            print(similarity)

            similarities.append(similarity)

        return similarities

    def compare_candidates_answers_with_fixed_answers(self, summarized_response, fixed_answers):
        # Analyze similarity with fixed answers
        similarities = self.analyze_similarity_using_spacy(summarized_response, fixed_answers)
        max_similarity = max(similarities)
        max_index = similarities.index(max_similarity)

        print(f"Similarities: {similarities}")
        print(f"max_similarity: {max_similarity}")
        print(f"max_index: {max_index}")
        print(f"Correct answer based on the similarity score: {fixed_answers[max_index]}")