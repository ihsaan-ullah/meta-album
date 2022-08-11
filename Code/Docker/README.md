
If you want to use docker, please pull the following docker image from DockerHub:

```bash
docker pull sunhaozhe/pytorch:1.1
```

When the docker image is ready, run the following two commands to create a docker container and enter it:


```bash
docker run -dit --name my_env --ipc=host --gpus all [ImageID] 
docker exec -it my_env bash
```



