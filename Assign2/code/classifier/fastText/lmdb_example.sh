#!/usr/bin/env bash
#
# Copyright (c) 2016-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.
#

myshuf() {
  perl -MList::Util=shuffle -e 'print shuffle(<>);' "$@";
}

normalize_text() {
  tr '[:upper:]' '[:lower:]' | sed -e 's/^/__label__/g' | \
    sed -e "s/'/ ' /g" -e 's/"//g' -e 's/\./ \. /g' -e 's/<br \/>/ /g' \
        -e 's/,/ , /g' -e 's/(/ ( /g' -e 's/)/ ) /g' -e 's/\!/ \! /g' \
        -e 's/\?/ \? /g' -e 's/\;/ /g' -e 's/\:/ /g' | tr -s " " | myshuf
}

RESULTDIR=lmdbresult
DATADIR=lmdbdata

mkdir -p "${RESULTDIR}"
mkdir -p "${DATADIR}"

if [ ! -f "${DATADIR}/lmdb.train" ]
then
  cat "${DATADIR}/output_train" | normalize_text > "${DATADIR}/lmdb.train"
  cat "${DATADIR}/output_test" | normalize_text > "${DATADIR}/lmdb.test"
fi

make

./fasttext supervised -input "${DATADIR}/lmdb.train" -output "${RESULTDIR}/lmdb" -dim 10 -lr 0.1 -wordNgrams 2 -minCount 1 -bucket 10000000 -epoch 5 -thread 4

./fasttext test "${RESULTDIR}/lmdb.bin" "${DATADIR}/lmdb.test"

./fasttext predict "${RESULTDIR}/lmdb.bin" "${DATADIR}/lmdb.test" > "${RESULTDIR}/lmdb.test.predict"
