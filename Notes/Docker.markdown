---
title: Docker
nav_order: 17
parent: Notes
layout: default
---

# Docker Guidebook

This guidebook provides a comprehensive introduction to Docker, a platform for containerizing and managing applications. It covers the basics, installation, core concepts, and practical examples for beginners and intermediate users.

## Table of Contents
1. [What is Docker?](#what-is-docker)
2. [Why Use Docker?](#why-use-docker)
3. [Installing Docker](#installing-docker)
4. [Docker Core Concepts](#docker-core-concepts)
5. [Basic Docker Commands](#basic-docker-commands)
6. [Creating a Dockerfile](#creating-a-dockerfile)
7. [Docker Compose](#docker-compose)
8. [Best Practices](#best-practices)
9. [Troubleshooting](#troubleshooting)
10. [Resources](#resources)

---

## What is Docker?

Docker is an open-source platform that uses containerization to package applications and their dependencies into isolated, lightweight containers. Containers allow developers to ensure consistent environments across development, testing, and production.

- **Containers vs. Virtual Machines**: Containers are more lightweight than VMs because they share the host OS kernel, reducing overhead.
- **Portability**: Containers run consistently on any system with Docker installed.
- **Microservices**: Docker is widely used in microservices architectures for deploying isolated services.

---

## Why Use Docker?

- **Consistency**: Eliminates "it works on my machine" issues by ensuring identical environments.
- **Efficiency**: Containers are lightweight, using fewer resources than VMs.
- **Scalability**: Easily scale applications by running multiple containers.
- **Isolation**: Each container runs in its own environment, preventing conflicts.
- **CI/CD Integration**: Simplifies continuous integration and deployment workflows.

---

## Installing Docker

Docker can be installed on various operating systems. Below are instructions for common platforms.

### On Ubuntu
1. Update the package index:
   ```bash
   sudo apt-get update
   ```
2. Install required packages:
   ```bash
   sudo apt-get install -y apt-transport-https ca-certificates curl software-properties-common
   ```
3. Add Docker’s official GPG key:
   ```bash
   curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
   ```
4. Add the Docker repository:
   ```bash
   sudo add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable"
   ```
5. Install Docker:
   ```bash
   sudo apt-get update
   sudo apt-get install -y docker-ce
   ```
6. Verify installation:
   ```bash
   sudo systemctl status docker
   docker --version
   ```

### On macOS
1. Download Docker Desktop from [Docker Hub](https://www.docker.com/products/docker-desktop).
2. Install the `.dmg` file by dragging it to the Applications folder.
3. Launch Docker Desktop and follow the setup wizard.
4. Verify installation:
   ```bash
   docker --version
   ```

### On Windows
1. Download Docker Desktop for Windows from [Docker Hub](https://www.docker.com/products/docker-desktop).
2. Ensure WSL 2 (Windows Subsystem for Linux 2) is enabled.
3. Install Docker Desktop and follow the setup wizard.
4. Verify installation:
   ```bash
   docker --version
   ```

### Post-Installation
- Add your user to the `docker` group to run Docker without `sudo` (Linux only):
  ```bash
  sudo usermod -aG docker $USER
  ```
- Test Docker with a sample container:
  ```bash
  docker run hello-world
  ```

---

## Docker Core Concepts

- **Image**: A read-only template used to create containers. Images are built from a `Dockerfile` or pulled from a registry like Docker Hub.
- **Container**: A running instance of an image. Containers are isolated but share the host OS.
- **Dockerfile**: A script containing instructions to build a Docker image.
- **Registry**: A storage and distribution system for Docker images (e.g., Docker Hub, AWS ECR).
- **Volume**: Persistent storage for containers, allowing data to persist beyond a container’s lifecycle.
- **Network**: Defines how containers communicate with each other and the outside world.

---

## Basic Docker Commands

Here are essential commands to get started with Docker.

### Image Commands
- Pull an image from a registry:
  ```bash
  docker pull <image-name>:<tag>
  ```
  Example: `docker pull nginx:latest`
- List local images:
  ```bash
  docker images
  ```
- Remove an image:
  ```bash
  docker rmi <image-name>:<tag>
  ```

### Container Commands
- Run a container:
  ```bash
  docker run -d --name <container-name> -p <host-port>:<container-port> <image-name>
  ```
  Example: `docker run -d --name my-nginx -p 8080:80 nginx`
- List running containers:
  ```bash
  docker ps
  ```
- List all containers (including stopped):
  ```bash
  docker ps -a
  ```
- Stop a container:
  ```bash
  docker stop <container-name>
  ```
- Remove a container:
  ```bash
  docker rm <container-name>
  ```
- View container logs:
  ```bash
  docker logs <container-name>
  ```

### System Commands
- Check Docker version:
  ```bash
  docker --version
  ```
- View system-wide information:
  ```bash
  docker info
  ```
- Clean up unused containers, images, and networks:
  ```bash
  docker system prune
  ```

---

## Creating a Dockerfile

A `Dockerfile` defines how to build a Docker image. Below is an example for a simple Node.js application.

### Example: Node.js App Dockerfile
```dockerfile
# Use an official Node.js runtime as the base image
FROM node:16

# Set the working directory inside the container
WORKDIR /app

# Copy package.json and install dependencies
COPY package.json .
RUN npm install

# Copy the rest of the application code
COPY . .

# Expose the application port
EXPOSE 3000

# Command to run the application
CMD ["npm", "start"]
```

### Steps to Build and Run
1. Create a `Dockerfile` in your project directory.
2. Build the image:
   ```bash
   docker build -t my-node-app .
   ```
3. Run the container:
   ```bash
   docker run -d -p 3000:3000 my-node-app
   ```
4. Access the app at `http://localhost:3000`.

---

## Docker Compose

Docker Compose is a tool for defining and running multi-container applications using a YAML file.

### Example: `docker-compose.yml`
```yaml
version: '3'
services:
  web:
    image: nginx:latest
    ports:
      - "8080:80"
    volumes:
      - ./html:/usr/share/nginx/html
  db:
    image: mysql:5.7
    environment:
      MYSQL_ROOT_PASSWORD: example
    volumes:
      - db-data:/var/lib/mysql
volumes:
  db-data:
```

### Commands
- Start services:
  ```bash
  docker-compose up -d
  ```
- Stop services:
  ```bash
  docker-compose down
  ```
- View running services:
  ```bash
  docker-compose ps
  ```

---

## Best Practices

- **Use Official Images**: Start with trusted base images from Docker Hub (e.g., `python:3.9`, `node:16`).
- **Minimize Image Size**: Use multi-stage builds and `.dockerignore` to exclude unnecessary files.
- **Avoid Running as Root**: Use `USER` in `Dockerfile` to run containers as a non-root user.
- **Leverage Caching**: Order `Dockerfile` instructions to maximize layer caching (e.g., copy `package.json` before source code).
- **Use Volumes for Persistence**: Store data outside containers to avoid data loss.
- **Tag Images Properly**: Use meaningful tags (e.g., `my-app:1.0.0`) instead of `latest`.
- **Monitor Resource Usage**: Use `docker stats` to monitor container performance.

---

## Troubleshooting

- **Container Exits Immediately**: Check logs with `docker logs <container-name>` to diagnose errors.
- **Port Already in Use**: Ensure the host port is free or use a different port.
- **Permission Denied**: Verify your user is in the `docker` group or use `sudo`.
- **Image Pull Fails**: Check internet connectivity or try a different registry.
- **Out of Disk Space**: Run `docker system prune` to clean up unused resources.

---

## Resources

- [Official Docker Documentation](https://docs.docker.com/)
- [Docker Hub](https://hub.docker.com/)
- [Docker Compose Reference](https://docs.docker.com/compose/)
- [Docker Best Practices](https://docs.docker.com/develop/develop-images/dockerfile_best-practices/)
- [Docker Community Forums](https://forums.docker.com/)

This guidebook provides a foundation for using Docker effectively. For advanced topics like Kubernetes integration or CI/CD pipelines, refer to the official documentation.

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