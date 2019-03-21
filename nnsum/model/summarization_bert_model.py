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

        batch = input.documents
        input_ids = torch.tensor(
            [f.input_ids for f in batch], dtype=torch.long)
        input_mask = torch.tensor(
            [f.input_mask for f in batch], dtype=torch.long)
        segment_ids = torch.tensor(
            [f.segment_ids for f in batch], dtype=torch.long)

        _, encoded_document = self.sentence_encoder(
            input_ids, segment_ids, input_mask)

        logits_and_attention = self.sentence_extractor(
            encoded_document,
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
