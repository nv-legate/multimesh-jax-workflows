#! /usr/bin/bash

/opt/scripts/devcontainer/install-build-deps.sh


BAZEL_CACHE=grpc://host.docker.internal:9092
BUILD_TYPE=${1:-Release}
BUILD_DIR=/opt/build
LIB_DIR=/opt/lib

# map the git submodule worktrees to the correct path
mkdir -p /docker
ln -s /opt/workspace /docker/workspace

# git finds what I'm doing 'dubious'
git config --global --add safe.directory /opt/workspace/xla
git config --global --add safe.directory /opt/workspace/multimesh-jax
git config --global --add safe.directory /opt/workspace/realm
git config --global --add safe.directory /opt/workspace/zuku

pushd /opt/workspace/realm
/opt/scripts/realm/configure.sh ${BUILD_TYPE}
/opt/scripts/realm/build.sh
git config remote.origin.fetch "+refs/heads/*:refs/remotes/origin/*"
popd

pushd /opt/workspace/zuku
/opt/scripts/zuku/configure.sh ${BUILD_TYPE}
/opt/scripts/zuku/build.sh
git config remote.origin.fetch "+refs/heads/*:refs/remotes/origin/*"
popd

pushd /opt/workspace/xla
/opt/scripts/multimesh-jax/configure-xla.sh
git config remote.origin.fetch "+refs/heads/*:refs/remotes/origin/*"
popd

pushd /opt/workspace/multimesh-jax
/opt/scripts/multimesh-jax/configure.sh ${BUILD_TYPE} ${BAZEL_CACHE}
/opt/scripts/multimesh-jax/build.sh
/opt/scripts/multimesh-jax/install.sh
git config remote.origin.fetch "+refs/heads/*:refs/remotes/origin/*"
popd

cd ~/
wget https://raw.githubusercontent.com/git/git/master/contrib/completion/git-completion.bash
# set up git completion
echo "source git-completion.bash" >> .bashrc
# ignore annoying lldb error
echo "settings set target.disable-aslr false" >> .lldbinit

pushd /opt/workspace/xla
#/opt/xla-compile-commands.sh
popd
