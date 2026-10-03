#!/usr/bin/env bash
set +e
set +u
set +o pipefail

mkdir -p data
cd data

mkdir -p librispeech_raw
cd librispeech_raw
wget -nc http://www.openslr.org/resources/12/dev-clean.tar.gz
tar -xzf dev-clean.tar.gz
cd ..

wget -nc http://download.magenta.tensorflow.org/datasets/nsynth/nsynth-valid.jsonwav.tar.gz
tar -xzf nsynth-valid.jsonwav.tar.gz
