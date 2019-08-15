import sys, math, argparse, re
from khaiii import KhaiiiApi
import mecab

def khaiii_tokenize(corpus_fname, output_fname):
    api = KhaiiiApi()

    with open(corpus_fname, 'r', encoding='utf-8') as f1, \
            open(output_fname, 'w', encoding='utf-8') as f2:
        for line in f1:
            sentence = line.replace('\n', '').strip()
            tokens = api.analyze(sentence)
            tokenized_sent = ''
            for token in tokens:
                tokenized_sent += ' '.join([str(m) for m in token.morphs]) + ' '
            f2.writelines(tokenized_sent.strip() + '\n')


def mecab_tokenize(corpus_fname, output_fname):
    mcab = mecab.MeCab()

    with open(corpus_fname, 'r', encoding='utf-8') as f1, \
            open(output_fname, 'w', encoding='utf-8') as f2:
        for line in f1:
            sentence = line.replace('\n', '').strip()
            tokens = mcab.morphs(sentence)
            tokenized_sent = ' '.join(tokens)
            f2.writelines(tokenized_sent + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--preprocess_mode', type=str, help='preprocess mode')
    parser.add_argument('--input_path', type=str, help='Location of input files')
    parser.add_argument('--output_path', type=str, help='Location of output files')
    args = parser.parse_args()

    if args.preprocess_mode == "khaiii_tokenize":
        khaiii_tokenize(args.input_path, args.output_path)
    elif args.preprocess_mode == "mecab_tokenize":
        mecab_tokenize(args.input_path, args.output_path)