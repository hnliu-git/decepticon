# Python Import:
from argparse import ArgumentParser

from transformers import (
    T5ForConditionalGeneration, T5Config, AutoTokenizer
)

# Pytorch Lightning Import:
import pytorch_lightning as pl


class RaceInfModule(pl.LightningModule):
    """ T5 Model """

    @staticmethod
    def add_model_specific_args(parent_parser):
        """"""
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--version", type=float,
                            help="specify it in a form X.XX")
        parser.add_argument("--padding_token", type=int, default=0,
                            help="don't change it")
        parser.add_argument("--tokenizer_len", type=int, default=32104,
                            help="don't touch it")
        parser.add_argument("--seed", default=1234, type=float)
        parser.add_argument("--weight_decay", default=5e-5, type=float)
        parser.add_argument("--learning_rate", default=1e-4, type=float)

        return parser

    def __init__(self, hparams):
        """
        :param batch_fn: function to process batch
        """
        super(RaceInfModule, self).__init__()

        self.hparams = hparams
        self.save_hyperparameters(hparams)

        if self.hparams.pretrained_model in ["t5-base","t5-small"]:
            # Model:
            config = T5Config(decoder_start_token_id=self.hparams.padding_token)
            self.model = T5ForConditionalGeneration(config).from_pretrained(self.hparams.pretrained_model)
            # Tokenizer:
            self.tokenizer = AutoTokenizer.from_pretrained(self.hparams.pretrained_model)
            self.tokenizer.add_special_tokens({"additional_special_tokens": ["[CON]","[QUE]","[ANS]","[DIS]"]})

            try:
                self.model.resize_token_embeddings(self.hparams.tokenizer_len)
            except:
                self.model.resize_token_embeddings(32104)
        else:
            raise NotImplementedError

    def generate(self, inputs, use_beam=False, use_sample=True, **kwargs):
        """ Args:
            inputs dict: dict of input
            kwargs: for generation

            Returns:
                id_seqs (bsz, pred_len)
        """
        assert use_beam or use_sample, 'Must use one method for generation'
        if use_beam:
            return self.generate_with_beam(inputs, **kwargs)
        if use_sample:
            return self.generate_with_sampling(inputs, **kwargs)

    def generate_with_beam(self, inputs,
                           num_beams: int = 6,
                           no_repeat_ngram_size: int = 2,
                           max_length: int = 64,
                           early_stopping: bool = True,
                           num_beam_groups: int = 2):
        """"""

        generated = self.model.generate(**inputs, # context -> answer + article
                                        num_beams=num_beams,
                                        num_beam_groups=num_beam_groups,
                                        max_length=max_length,
                                        no_repeat_ngram_size=no_repeat_ngram_size,
                                        early_stopping=early_stopping)

        return generated

    def generate_with_sampling(self, inputs,
                               top_k: int = 50, ##1 75
                               top_p: float = 0.95, ##2 0.9
                               max_length: int = 64,
                               do_sample: bool = True,
                               no_repeat_ngram_size: int = 2,
                               num_samples = 1):
        """"""
        # [bsz, pred_len]
        try:
            top_k = self.top_k ##1 75
            top_p = self.top_p ##2 0.9
            no_repeat_ngram_size = self.no_repeat_ngram_size
            num_samples = self.num_samples
        except:
            pass
        
        generated = self.model.generate(**inputs, # context -> answer + article
                                        max_length=max_length,
                                        do_sample=do_sample,
                                        no_repeat_ngram_size=no_repeat_ngram_size,
                                        num_return_sequences=num_samples,
                                        top_k=top_k,
                                        top_p=top_p)
        return generated

    def generate_sentence(self, article, answer, question=None):
        """Args:
            article (str)
            answer (str)
            question (str): if not none, generating distractors
            skip_special_tokens (bool): skip special_tokens while decoding
        :return:
            list of generated sentences, len(list) = sample_num
        """
        if question:
            context = " ".join(['[ANS]', answer, '[QUE]', question, '[CON]', article])
        else:
            context = " ".join(['[ANS]', answer, '[CON]', article])
        inputs = self.tokenizer([context], padding=True, truncation=True, max_length=512, return_tensors="pt")
        sentence = self.generate(inputs, use_sample=True)

        return self.tokenizer.decode(sentence.squeeze(), skip_special_tokens=True)