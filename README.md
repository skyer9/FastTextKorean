# EmbedRankKorean

## 설치

**아래 메뉴얼은 AWS EC2 r4.large 인스턴스에서 테스트되었습니다.**

Embedding 에 대한 기초지식은 [여기](https://ratsgo.github.io/embedding/) 가 좋다. 인터넷 전체를 구글링하는것보다 여기 블로그를 찾는 것이 낫다.

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

fasttext 를 컴파일한다.

``` sh
cd ~/
git clone https://github.com/facebookresearch/fastText.git
cd fastText
make
```

python 모듈을 설치한다.

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

[구글 드라이브](https://drive.google.com/file/d/1kUecR7xO7bsHFmUI6AExtY5u2XXlObOG/view) 에서 전처리가 끝난 데이타를 다운받는다.

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

모델을 학습하고, 데이타를 형태소분석 한다.

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
```

### sent2vec 모델 생성

```sh
cd ~/
git clone https://github.com/epfml/sent2vec.git
cd sent2vec
make

mkdir mywork
cp ~/embedding/mywork/ratings_tokenized.txt ./mywork/
cd mywork
../fasttext sent2vec \
                    -input ./ratings_tokenized.txt \
                    -output model \
                    -minCount 8 \
                    -dim 700 \
                    -epoch 9 \
                    -lr 0.2 \
                    -wordNgrams 2 \
                    -loss ns \
                    -neg 10 \
                    -thread 20 \
                    -t 0.000005 \
                    -dropoutK 4 \
                    -minCountLabel 20 \
                    -bucket 4000000 \
                    -maxVocabSize 750000 \
                    -numCheckPoints 10
```

### khaiii 설치

[문서](https://github.com/kakao/khaiii/wiki/%EB%B9%8C%EB%93%9C-%EB%B0%8F-%EC%84%A4%EC%B9%98) 를 참조하여 khaiii 를 설치한다.

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

## 사용법

### fasttext 사용법

데이타 준비

```sh
cd ~/fastText
mkdir mywork
cp ~/embedding/mywork/processed/corrected_ratings_corpus.txt mywork/
```

```sh
cd mywork

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

fasttext 는 후기 평점 예측, 고객센터 자동응답, 그리고 멀티 라벨링도 지원하기에 상품 속성 추출에도 적용할만 하다.

보다 많은 자료는 [여기](https://github.com/facebookresearch/fastText/tree/master/docs) 에서 확인할 수 있다.
