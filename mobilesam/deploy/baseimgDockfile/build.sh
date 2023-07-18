docker build -t mobilesam_base:1.0.1 .
docker tag mobilesam_base:1.0.1  registry.cn-hangzhou.aliyuncs.com/deep_learning_wwl/deeplearning:1.0.1
docker push registry.cn-hangzhou.aliyuncs.com/deep_learning_wwl/deeplearning:1.0.1