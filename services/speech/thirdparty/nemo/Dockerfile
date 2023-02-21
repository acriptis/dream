FROM pytorch/pytorch:1.4-cuda10.1-cudnn7-devel

WORKDIR /app

RUN rm -r /etc/apt/sources.list.d && \
    apt update && apt install -y libsndfile-dev

# WeNet
# Source of WeNet Dockerfile: https://github.com/wenet-e2e/wenet/blob/main/runtime/libtorch/docker/Dockerfile
RUN sed -i s@/archive.ubuntu.com/@/mirrors.tuna.tsinghua.edu.cn/@g /etc/apt/sources.list && \
    apt-get update && apt-get install -y git wget build-essential libssl-dev
# Installing latest cmake (If I installed it with apt, the version would be not recent enough.)
# Source doc: https://graspingtech.com/upgrade-cmake/
RUN cd /home && \
    wget https://github.com/Kitware/CMake/releases/download/v3.20.2/cmake-3.20.2.tar.gz && \
    tar -zxvf cmake-3.20.2.tar.gz && \
    cd cmake-3.20.2 && \
    ./bootstrap && \
    make && \
    make install
# Installing WeNet
# (With eliminating symlinks by copying the content. (It's needed because cmake fails when it arrives at the wrong directory by stepping to .. from a symlink directory.))
COPY grpc_client_main.cc .
RUN git clone https://github.com/wenet-e2e/wenet.git /home/wenet && \
    build=/home/wenet/runtime/libtorch && \
    mv grpc_client_main.cc $build/../core/bin && \
    cd $build && \
    for d in `find . -type l` ; do source=`readlink $d`; rm $d; cp -r $source . ; done && \
    cmake -B $build -DCMAKE_BUILD_TYPE=Release -DGRAPH_TOOLS=ON -DGRPC=ON && \
    cmake --build $build --parallel


# Dream
COPY requirements.txt tts.json ./

RUN pip install deeppavlov==0.12.1 && \
    python -m deeppavlov install asr_tts

RUN python -m deeppavlov download tts.json

RUN pip install -r requirements.txt

# WeNet ASR model
RUN mkdir /app/asr && \
	wget -O- https://wenet-1256283475.cos.ap-shanghai.myqcloud.com/models/gigaspeech/gigaspeech_u2pp_conformer_libtorch.tar.gz | tar -xz -C asr && \
	model_dir=`ls asr` && \
	ln -s $PWD/asr/$model_dir/final.zip asr && \
	ln -s $PWD/asr/$model_dir/units.txt asr


COPY main.py .
COPY start.sh .

#ENTRYPOINT /app/start.sh 
