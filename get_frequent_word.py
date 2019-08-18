import sys, math, argparse, re
import mecab
from soynlp.word import WordExtractor
from soynlp.tokenizer import LTokenizer
from soynlp.normalizer import *
from soyspacing.countbase import CountSpace
from soynlp.hangle import decompose, character_is_korean
import hgtk


def word_count(corpus_fname):
    with open(corpus_fname, 'r', encoding='utf-8') as f:
        sentences = f.read()
        words = re.findall("[가-힣]+", sentences)

        # print(words)
        d = {}
        for word in words:
            d[word] = d.get(word, 0) + 1

        # print(d)
        word_freq = []
        for key, value in d.items():
            word_freq.append((value, key))

        # print(word_freq)
        word_freq.sort(reverse=True)
        return word_freq


def is_all_nng(words):
    # [('자연주의', 'NNG'), ('쇼핑몰', 'NNG')]
    for item in words:
        (w, p) = item
        if p != 'NNG':
            return False
    return True


def check_morphs(lst, corpus_fname, output_fname, log_fname):
    mcab = mecab.MeCab()

    model_fname = 'soyword.model'
    word_extractor = WordExtractor(
        min_frequency=100,
        min_cohesion_forward=0.05,
        min_right_branching_entropy=0.0
    )
    word_extractor.load(model_fname)
    scores = word_extractor.word_scores()
    scores = {key:(scores[key].cohesion_forward * math.exp(scores[key].right_branching_entropy)) for key in scores.keys()}
    soy_tokenizer = LTokenizer(scores=scores)

    with open(corpus_fname, 'r', encoding='utf-8') as f1, \
         open(output_fname, 'w', encoding='utf-8') as f2, \
         open(log_fname, 'w', encoding='utf-8') as f3:
        sentences = f1.read()

        for item in lst:
            cnt, word = item

            if cnt < 100 or len(word) == 1:
                continue

            tokens = mcab.morphs(word)
            if len(tokens) == 1:
                continue

            (cho, jung, jong) = hgtk.letter.decompose(word[-1])
            if 'ㄱ' <= jong <= 'ㅎ':
                dic_line = "{},,,,NNP,*,{},{},*,*,*,*,*".format(word, 'T', word)
            else:
                dic_line = "{},,,,NNP,*,{},{},*,*,*,*,*".format(word, 'F', word)
            f2.writelines(dic_line + '\n')
            f3.writelines("{}\t{}\t{}".format(word, ' '.join(tokens), cnt) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, help='Location of input files')
    parser.add_argument('--output_path', type=str, help='Location of output files')
    parser.add_argument('--log_path', type=str, help='Location of log files')
    args = parser.parse_args()

    lst = word_count(args.input_path)
    # print(lst)

    # for item in lst:
    #     cnt, word = item

    #     if cnt >= 100 and len(word) > 1:
    #         print("{}\t{}".format(word, cnt))

    check_morphs(lst, args.input_path, args.output_path, args.log_path)
