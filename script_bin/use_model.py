import argparse
import pathlib
import ujson as json

from multiprocessing import cpu_count
import torch
from tqdm import tqdm

import nnsum
from pytorch_pretrained_bert.tokenization import BertTokenizer


def main():
    parser = argparse.ArgumentParser(
        "Use an nnsum model with Bert to create summaries")
    parser.add_argument("--gpu", default=-1, type=int)
    parser.add_argument("--sentence-limit", default=None, type=int)
    parser.add_argument("--summary-length", type=int, default=100)
    parser.add_argument("--loader-workers", type=int, default=None)
    parser.add_argument(
        "--remove-stopwords", action="store_true", default=False)
    parser.add_argument(
        "--input", type=str, required=True)
    parser.add_argument(
        "--model", type=pathlib.Path, required=True)
    parser.add_argument("--max_seq_length", default=128, type=int)
    parser.add_argument(
        "--bert", type=pathlib.Path, required=True)

    args = parser.parse_args()

    if args.loader_workers is None:
        args.loader_workers = min(16, cpu_count())

    model = torch.load(args.model, map_location=lambda storage, loc: storage)
    if args.gpu > -1:
        model.cuda(args.gpu)

    tokenizer = BertTokenizer.from_pretrained(args.bert)
    model.eval()
    with torch.no_grad():
        for line in tqdm(open(args.input)):
            d = json.loads(line)
            data = nnsum.data.SummarizationDatasetForTagging(
                tokenizer,
                args.max_seq_length,
                d,
                sentence_limit=args.sentence_limit)

            loader = nnsum.data.SummarizationDataLoaderForBert(
                data, batch_size=1,
                num_workers=args.loader_workers)

            for step, batch in enumerate(loader):
                assert step == 0
                batch = batch.to(args.gpu)
                outputs = model.predict(batch, max_length=args.summary_length)
                assert len(outputs) == 1
                d['nnsum'] = outputs[0]

            print(json.dumps(d))


if __name__ == "__main__":
    main()
