# EmbedRankKorean

## 설치

**아래 메뉴얼은 AWS EC2 r4.large 인스턴스에서 테스트되었습니다.**

Embedding 에 대한 기초지식은 [여기](https://ratsgo.github.io/embedding/) 가 좋습니다. 인터넷 전체를 구글링하는것보다 여기 블로그를 찾는 것이 낫습니다.

### 개발환경 구성

```sh
sudo apt-get update
sudo apt-get upgrade
sudo apt-get install python-setuptools
sudo apt install python3-pip

pip3 install soynlp
pip3 install soyspacing
pip3 install sentencepiece
pip3 install bert
pip3 install bert-tensorflow
pip3 install tensorflow

sudo apt install make
sudo apt-get install build-essential

sudo apt install unzip

sudo apt install cmake
```

### fasttext 설치

fasttext 를 컴파일합니다.

``` sh
cd ~/
git clone https://github.com/facebookresearch/fastText.git
cd fastText
make
```

python 모듈을 설치합니다.

```sh
pip3 install .
```

```sh
mkdir ~/bin
cp fasttext ~/bin/
~/bin/fasttext -help
```

### 데이타 전처리

#### 전처리 끝난 데이타 다운받기

[여기](https://github.com/ratsgo/embedding) 를 클론한다.

```sh
cd ~/
git clone https://github.com/ratsgo/embedding.git
cd embedding
```

[구글 드라이브](https://drive.google.com/file/d/1kUecR7xO7bsHFmUI6AExtY5u2XXlObOG/view) 에서 전처리가 끝난 데이타를 다운받습니다.

```sh
vi ~/bin/gdrive_download

-----------------------------------------
#!/usr/bin/env bash

# gdrive_download
#
# script to download Google Drive files from command line
# not guaranteed to work indefinitely
# taken from Stack Overflow answer:
# http://stackoverflow.com/a/38937732/7002068

gURL=$1
# match more than 26 word characters
ggID=$(echo "$gURL" | egrep -o '(\w|-){26,}')

ggURL='https://drive.google.com/uc?export=download'

curl -sc /tmp/gcokie "${ggURL}&id=${ggID}" >/dev/null
getcode="$(awk '/_warning_/ {print $NF}' /tmp/gcokie)"

cmd='curl --insecure -C - -LOJb /tmp/gcokie "${ggURL}&confirm=${getcode}&id=${ggID}"'
echo -e "Downloading from "$gURL"...\n"
eval $cmd
-----------------------------------------

chmod 700 ~/bin/gdrive_download

~/bin/gdrive_download https://drive.google.com/file/d/1kUecR7xO7bsHFmUI6AExtY5u2XXlObOG
```

모델을 학습하고, 데이타를 형태소분석 합니다.

```sh
mkdir mywork
cd mywork
mv ../processed.zip ./
unzip processed.zip

# train
python3 ../preprocess/unsupervised_nlputils.py --preprocess_mode compute_soy_word_score \
    --input_path ./processed/corrected_ratings_corpus.txt \
    --model_path ./soyword.model

# tokenize
python3 ../preprocess/unsupervised_nlputils.py --preprocess_mode soy_tokenize \
    --input_path ./processed/corrected_ratings_corpus.txt \
    --model_path ./soyword.model \
    --output_path ./ratings_tokenized.txt

head -5 ./ratings_tokenized.txt
어릴때 보고 지금 다시 봐도 재밌 어ㅋㅋ
디자인을 배우 는 학생 으로, 외국 디자이너와 그들 이 일군 전통을 통해 발전 해가는 문화 산업이 부러웠는데. 사실 우리나라 에서도 그 어려운 시절에 끝까지 열정 을 지킨 노라노 같은 전통이있어 저와 같은 사람들이 꿈을 꾸고 이뤄나갈 수 있다 는 것에 감사합니다.
폴리스스토리 시리즈는 1부터 뉴까지 버릴께 하나 도 없음. . 최고 .
와.. 연기 가 진짜 개쩔구나.. 지루 할거라고 생각 했는데 몰입 해서 봤다. . 그래 이런 게 진짜 영화 지
```

### khaiii 설치

[문서](https://github.com/kakao/khaiii/wiki/%EB%B9%8C%EB%93%9C-%EB%B0%8F-%EC%84%A4%EC%B9%98) 를 참조하여 khaiii 를 설치합니다.

```sh
cd ~/
git clone https://github.com/kakao/khaiii.git
cd khaiii
cmake --version

mkdir build
cd build/
cmake ..

make all
make resource
make large_resource
sudo make install
khaiii --help
```

테스트하기

```sh
vi input.txt
-----------------------------------------
동해물과 백두산이 마르고 닳도록 하느님이 보우하사 우리나라 만세
무궁화 삼천리 화려강산 대한 사람 대한으로 길이 보전하세
-----------------------------------------

khaiii --input input.txt
```

아래 명령으로 python 연동모듈을 설치할 수 있습니다.

```sh
make package_python
cd package_python
pip3 install .
```

### mecab-ko 설치

```sh
# mecab-ko
wget https://bitbucket.org/eunjeon/mecab-ko/downloads/mecab-0.996-ko-0.9.2.tar.gz
tar xvfz mecab-0.996-ko-0.9.2.tar.gz
cd mecab-0.996-ko-0.9.2
./configure --prefix=/usr
make
make check
sudo make install

# mecab-ko-dic
sudo ldconfig
ldconfig -p | grep /usr/local/lib
cd ~/
wget https://bitbucket.org/eunjeon/mecab-ko-dic/downloads/mecab-ko-dic-2.1.1-20180720.tar.gz
tar xvfz mecab-ko-dic-2.1.1-20180720.tar.gz
cd mecab-ko-dic-2.1.1-20180720
./configure --prefix=/usr
make
sudo make install

# mecab-python
pip3 install python-mecab-ko
```

## 사용법

### fasttext 사용법

#### 데이타 준비

```sh
cd ~/fastText
mkdir mywork
cp ~/embedding/mywork/processed/corrected_ratings_corpus.txt mywork/
```

```sh
cd mywork

#### 형태소분석 없는 데이타로 학습

# using cbow
../fasttext cbow -input corrected_ratings_corpus.txt -output model_cbow

# using skipgram
../fasttext skipgram -input corrected_ratings_corpus.txt -output model_skipgram

# print vector
echo “디즈니” | ../fasttext print-word-vectors model_skipgram.bin

# nearest neighbors
echo “디즈니” | ../fasttext nn model_skipgram.bin
```

Text Classification 을 위한 학습데이타를 다운받는다.

```sh
wget https://dl.fbaipublicfiles.com/fasttext/data/cooking.stackexchange.tar.gz
tar xvzf cooking.stackexchange.tar.gz
head cooking.stackexchange.txt

# 데이타셋 분리
head -n 12404 cooking.stackexchange.txt > cooking.train
tail -n 3000 cooking.stackexchange.txt > cooking.test
```

```sh
../fasttext supervised -input cooking.train -output model_cooking
```

```sh
../fasttext predict model_cooking.bin -
Which baking dish is best to bake a banana bread ?
__label__baking
^C

../fasttext predict model_cooking.bin - 5
Why not put knives in the dishwasher?
__label__food-safety __label__baking __label__bread __label__equipment __label__substitutions
```

파라미터 정리

| parameter         | description                                      | default   |
|-------------------|--------------------------------------------------|-----------|
| input             | training file path                               | mandatory |
| output            | output file path                                 | mandatory |
| verbose           | verbosity level                                  | 2         |
| minCount          | minimal number of word occurences                | 5         |
| minCountLabel     | minimal number of label occurences               | 0         |
| wordNgrams        | max length of word ngram                         | 1         |
| bucket            | number of buckets                                | 2000000   |
| minn              | min length of char ngram                         | 3         |
| maxn              | max length of char ngram                         | 6         |
| t                 | sampling threshold                               | 0.0001    |
| label             | labels prefix                                    | []        |
| lr                | learning rate                                    | 0.05      |
| lrUpdateRate      | change the rate of updates for the learning rate | 100       |
| dim               | size of word vectors                             | 100       |
| ws                | size of the context window                       | 5         |
| epoch             | number of epochs                                 | 5         |
| neg               | number of negatives sampled                      | 5         |
| loss              | loss function {ns, hs, softmax}                  | ns        |
| thread            | number of threads                                | 12        |
| pretrainedVectors | pretrained word vectors for supervised learning  | []        |
| saveOutput        | whether output params should be saved            | 0         |
| cutoff            | number of words and ngrams to retain             | 0         |
| retrain           | finetune embeddings if a cutoff is applied       | 0         |
| qnorm             | quantizing the norm separately                   | 0         |
| qout              | quantizing the classifier                        | 0         |
| dsub              | size of each sub-vector                          | 2         |

#### 형태소분석 진행한 데이타로 학습(soy_tokenize)

```sh
cd ~/fastText/mywork/
cp ~/embedding/mywork/ratings_tokenized.txt mywork/
```

```sh
cd mywork

#### 형태소분석 진행한 데이타로 학습
../fasttext skipgram -input ratings_tokenized.txt -output model_skipgram

# print vector
echo “디즈니” | ../fasttext print-word-vectors model_skipgram.bin

# nearest neighbors
echo “디즈니” | ../fasttext nn model_skipgram.bin
Query word? 디즈니 0.994381
픽사 0.720645
애니메이션 0.703237
애니중 0.690123
애니의 0.677468
웍스 0.676697
애니를 0.675644
애니 0.674097
매이션 0.662758
에니메이션 0.661454
```

#### 형태소분석 진행한 데이타로 학습(khaiii)

```sh
vi unsupervised_nlputils.py
```

```python
import sys, math, argparse, re
from khaiii import KhaiiiApi

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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--preprocess_mode', type=str, help='preprocess mode')
    parser.add_argument('--input_path', type=str, help='Location of input files')
    parser.add_argument('--output_path', type=str, help='Location of output files')
    args = parser.parse_args()

    if args.preprocess_mode == "khaiii_tokenize":
        khaiii_tokenize(args.input_path, args.output_path)
```

```sh
python3 ./unsupervised_nlputils.py --preprocess_mode khaiii_tokenize \
    --input_path ./corrected_ratings_corpus.txt \
    --output_path ./ratings_tokenized.txt

../fasttext skipgram -input ratings_tokenized.txt -output model_skipgram

echo “디즈니/NNP” | ../fasttext nn model_skipgram.bin
```

#### 형태소분석 진행한 데이타로 학습(mecab-ko)

```sh
vi unsupervised_nlputils.py
```

```python
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
```

```sh
python3 ./unsupervised_nlputils.py --preprocess_mode mecab_tokenize \
    --input_path ./corrected_ratings_corpus.txt \
    --output_path ./ratings_tokenized.txt

../fasttext skipgram -input ratings_tokenized.txt -output model_skipgram

echo “디즈니” | ../fasttext nn model_skipgram.bin
Query word? 디즈니 0.996321
픽사 0.784137
드림웍스 0.762146
타잔 0.759206
애니메 0.732718
애니 0.718813
월트 0.702987
애니메이션 0.702211
클레이 0.683714
지브리 0.67412
```

## 후기

fasttext 는 후기 평점 예측, 고객센터 자동응답, 그리고 멀티 라벨링도 지원하기에 상품 속성 추출에도 적용할만 합니다.

보다 많은 자료는 [여기](https://github.com/facebookresearch/fastText/tree/master/docs) 에서 확인할 수 있습니다.
