---
title: Docker
nav_order: 17
parent: Notes
layout: default
---

# Docker Cheatsheet

This cheatsheet provides a quick reference for common Docker commands and concepts, supplementing the Docker Guidebook. It’s designed for beginners and intermediate users to quickly find essential commands.

## Table of Contents
- [Docker Installation](#docker-installation)
- [Image Commands](#image-commands)
- [Container Commands](#container-commands)
- [Dockerfile Basics](#dockerfile-basics)
- [Docker Compose Commands](#docker-compose-commands)
- [Volume Commands](#volume-commands)
- [Network Commands](#network-commands)
- [System Commands](#system-commands)
- [Useful Tips](#useful-tips)

---

## Docker Installation

- **Install Docker (Ubuntu)**:
  ```bash
  sudo apt-get update
  sudo apt-get install -y docker-ce
  ```
- **Add user to Docker group** (Linux):
  ```bash
  sudo usermod -aG docker $USER
  ```
- **Test installation**:
  ```bash
  docker run hello-world
  ```

---

## Image Commands

- **Pull an image**:
  ```bash
  docker pull <image-name>:<tag>
  ```
  Example: `docker pull nginx:latest`
- **List images**:
  ```bash
  docker images
  ```
- **Build an image**:
  ```bash
  docker build -t <image-name>:<tag> .
  ```
- **Remove an image**:
  ```bash
  docker rmi <image-name>:<tag>
  ```
- **Tag an image**:
  ```bash
  docker tag <source-image>:<tag> <new-image>:<tag>
  ```
- **Push to registry**:
  ```bash
  docker push <image-name>:<tag>
  ```

---

## Container Commands

- **Run a container** (detached, with port mapping):
  ```bash
  docker run -d --name <container-name> -p <host-port>:<container-port> <image-name>
  ```
  Example: `docker run -d --name my-nginx -p 8080:80 nginx`
- **List running containers**:
  ```bash
  docker ps
  ```
- **List all containers** (including stopped):
  ```bash
  docker ps -a
  ```
- **Stop a container**:
  ```bash
  docker stop <container-name>
  ```
- **Start a container**:
  ```bash
  docker start <container-name>
  ```
- **Remove a container**:
  ```bash
  docker rm <container-name>
  ```
- **View container logs**:
  ```bash
  docker logs <container-name>
  ```
- **Execute command in container**:
  ```bash
  docker exec -it <container-name> <command>
  ```
  Example: `docker exec -it my-container bash`
- **Inspect container details**:
  ```bash
  docker inspect <container-name>
  ```

---

## Dockerfile Basics

- **Sample Dockerfile** (Node.js app):
  ```dockerfile
  FROM node:16
  WORKDIR /app
  COPY package.json .
  RUN npm install
  COPY . .
  EXPOSE 3000
  CMD ["npm", "start"]
  ```
- **Build image**:
  ```bash
  docker build -t my-app:latest .
  ```
- **Common Instructions**:
  - `FROM`: Base image
  - `WORKDIR`: Set working directory
  - `COPY`: Copy files to container
  - `RUN`: Execute command during build
  - `EXPOSE`: Document port
  - `CMD`: Default command to run

---

## Docker Compose Commands

- **Sample `docker-compose.yml`**:
  ```yaml
  version: '3'
  services:
    web:
      image: nginx:latest
      ports:
        - "8080:80"
    db:
      image: mysql:5.7
      environment:
        MYSQL_ROOT_PASSWORD: example
  ```
- **Start services**:
  ```bash
  docker-compose up -d
  ```
- **Stop services**:
  ```bash
  docker-compose down
  ```
- **List services**:
  ```bash
  docker-compose ps
  ```
- **View logs**:
  ```bash
  docker-compose logs
  ```
- **Build services**:
  ```bash
  docker-compose build
  ```

---

## Volume Commands

- **Create a volume**:
  ```bash
  docker volume create <volume-name>
  ```
- **List volumes**:
  ```bash
  docker volume ls
  ```
- **Inspect a volume**:
  ```bash
  docker volume inspect <volume-name>
  ```
- **Remove a volume**:
  ```bash
  docker volume rm <volume-name>
  ```
- **Run with volume**:
  ```bash
  docker run -v <volume-name>:<container-path> <image-name>
  ```

---

## Network Commands

- **List networks**:
  ```bash
  docker network ls
  ```
- **Create a network**:
  ```bash
  docker network create <network-name>
  ```
- **Connect container to network**:
  ```bash
  docker network connect <network-name> <container-name>
  ```
- **Inspect network**:
  ```bash
  docker network inspect <network-name>
  ```
- **Remove network**:
  ```bash
  docker network rm <network-name>
  ```

---

## System Commands

- **Check Docker version**:
  ```bash
  docker --version
  ```
- **View system info**:
  ```bash
  docker info
  ```
- **Clean up unused resources**:
  ```bash
  docker system prune
  ```
- **Remove all unused images**:
  ```bash
  docker image prune -a
  ```
- **Check container resource usage**:
  ```bash
  docker stats
  ```

---

## Useful Tips

- **Run container interactively**:
  ```bash
  docker run -it <image-name> bash
  ```
- **Copy files to/from container**:
  ```bash
  docker cp <container-name>:<path> <host-path>
  docker cp <host-path> <container-name>:<path>
  ```
- **Save image to file**:
  ```bash
  docker save -o <file.tar> <image-name>:<tag>
  ```
- **Load image from file**:
  ```bash
  docker load -i <file.tar>
  ```
- **Limit container resources**:
  ```bash
  docker run --memory="512m" --cpus="0.5" <image-name>
  ```
- **View Docker events**:
  ```bash
  docker events
  ```

This cheatsheet is a quick reference for Docker commands and concepts. For detailed explanations, refer to the [Docker Guidebook](#11d5b84f-1ba5-4d78-b4bf-d1f859527ba4) or the [official Docker documentation](https://docs.docker.com/).