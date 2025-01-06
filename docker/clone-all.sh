#! /usr/bin/env bash

to_clone=${1:-all}

function checkout {
  folder=$1
  branch=$2
  repo=$3
  commit=$4
  patch=$5
  if [[ "$to_clone" != "all" ]] && [[ "$folder" != "$to_clone" ]]; then
    echo "skip individual $folder"
    return 0
  fi
  if ! [ -d "$folder" ]; then
    echo "cloning $branch of $repo into $folder"
    git clone -b $branch --filter tree:0 $repo $folder
    pushd $folder
    if [ ! -z "$commit" ]; then
      git checkout $commit
    fi
    if [ ! -z "$patch" ]; then
      git apply ../$patch
    fi
    popd
  fi
}

checkout chex       master              https://github.com/google-deepmind/chex.git   a3f1dd0
checkout clu        main                https://github.com/google/CommonLoopUtils.git f30bc44
checkout flax       main                https://github.com/google/flax.git            5694517
checkout jax        legate-main         ssh://git@github.com/nv-legate/jax.git        86b02af7ab58
checkout jaxlib     legate-main         ssh://git@github.com/nv-legate/jax.git        86b02af7ab58
checkout zuku       main                ssh://git@gitlab-master.nvidia.com:12051/jwilke/realm-poc.git 73f4be6f80
checkout legate-jax realm-refactor      ssh://git@github.com/nv-legate/legate.jax.git b9d372404e60
checkout realm      master              https://gitlab.com/StanfordLegion/legion.git  6da07108
checkout optax      main                https://github.com/google-deepmind/optax.git  a49564e
checkout orbax      main                https://github.com/google/orbax.git           1e06498bfb
checkout paxml      main                https://github.com/google/paxml.git           bd0590f843   paxml.patch
checkout praxis     main                https://github.com/google/praxis.git          81154d8e       praxis.patch
checkout seqio      main                https://github.com/google/seqio.git           513d1fe
checkout te         main                https://github.com/NVIDIA/TransformerEngine.git  a68acd71d350 te.patch
checkout te-xla     realm-refactor      ssh://git@github.com/nv-legate/xla.git        8240d55ed
checkout xla        realm-refactor      ssh://git@github.com/nv-legate/xla.git        fb673b06d
checkout nccl       master              https://github.com/NVIDIA/nccl.git            6203b4c95d71
checkout maxtext    main         https://github.com/AI-Hypercomputer/maxtext.git 0a919c1991 maxtext.patch

pushd te
git submodule update --init --recursive
popd

cp ../maxtext/run.py maxtext
cp ../paxml/run.py paxml
