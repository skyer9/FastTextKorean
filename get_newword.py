import sys, math, argparse, re
import mecab
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


def check_morphs(lst, corpus_fname, output_fname, log_fname):
    mcab = mecab.MeCab()

    with open(corpus_fname, 'r', encoding='utf-8') as f1, \
         open(output_fname, 'w', encoding='utf-8') as f2, \
         open(log_fname, 'w', encoding='utf-8') as f3:
        sentences = f1.read()

        for item in lst:
            cnt, word = item

            if cnt < 10:
                continue
            tokens = mcab.morphs(word)
            if len(tokens) == 1:
                continue

            words = re.findall(' '.join(tokens), sentences)
            if len(words) < (cnt * 0.05):
                # 형태소 분리된 단어의 빈도수가 분리안된 단어의 빈수도의 5% 미만이면 형태소 분리오류
                (cho, jung, jong) = hgtk.letter.decompose(word[-1])
                if 'ㄱ' <= jong <= 'ㅎ':
                    dic_line = "{},,,,NNP,*,{},{},*,*,*,*,*".format(word, 'T', word)
                else:
                    dic_line = "{},,,,NNP,*,{},{},*,*,*,*,*".format(word, 'F', word)
                # print("{}\t{}\t{}\t{}\t{}".format(word, ' '.join(tokens), cnt, len(words), jong))
                f2.writelines(dic_line + '\n')
                f3.writelines("{}\t{}\t{}\t{}".format(word, ' '.join(tokens), cnt, len(words)) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, help='Location of input files')
    parser.add_argument('--output_path', type=str, help='Location of output files')
    parser.add_argument('--log_path', type=str, help='Location of log files')
    args = parser.parse_args()

    lst = word_count(args.input_path)
    # print(lst)

    check_morphs(lst, args.input_path, args.output_path, args.log_path)
