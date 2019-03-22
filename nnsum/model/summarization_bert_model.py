import torch

from nnsum.model import SummarizationModel

from pytorch_pretrained_bert.modeling import BertModel  # noqa


class SummarizationBertModel(SummarizationModel):
    def __init__(self, sentence_encoder, sentence_extractor):
        super(SummarizationModel, self).__init__()
        self.sentence_encoder = sentence_encoder
        self.sentence_extractor = sentence_extractor

    def initialize_parameters(self, logger=None):
        if logger:
            logger.info(" Model parameter initialization started.")
        for module in self.children():
            if not isinstance(module, BertModel):
                module.initialize_parameters(logger=logger)
        if logger:
            logger.info(" Model parameter initialization finished.\n")

    def forward(self, input, decoder_supervision=None, mask_logits=False,
                return_attention=False):

        batch_size, docs_size, seq_length = input.document.size()
        assert seq_length % 3 == 0
        seq_length //= 3

        encoded_docs = torch.LongTensor(batch_size, 28, 768).to(
            device=input.document.device)

        for c, doc in enumerate(input.document):
            input_ids, segment_ids, input_mask = doc.split(
                seq_length, dim=-1)
            print(
                "arg sizes:", input_ids.size(), segment_ids.size(),
                input_mask.size())
            _, encoded_doc = self.sentence_encoder(
                input_ids, segment_ids, input_mask)

            print('encoded size:', encoded_doc.size())

            encoded_docs[c] = encoded_doc

        logits_and_attention = self.sentence_extractor(
            encoded_docs,
            input.num_sentences,
            targets=decoder_supervision)

        if isinstance(logits_and_attention, (list, tuple)):
            logits, attention = logits_and_attention
        else:
            logits = logits_and_attention
            attention = None

        if return_attention:
            return logits, attention
        else:
            return logits
