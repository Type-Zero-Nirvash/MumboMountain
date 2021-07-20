#####################################################################
# To use this Dockerfile and see our results.
#
# You must pass "-s" or "--save" at runtime to save our graphs
# This can be done if running our examples outside of Docker as well

#####################################################################
# Ubuntu users:
# You MUST do one of the following to prevent pip3 from failing.

#####################################################################
# OPTION ONE
# Use the '--network=host' argument when building the image
#
# Example:
# * sudo may be optional *
# sudo docker build --network=host -t nirvash/mumbo .

#####################################################################
# OPTION TWO
#
# 1) 
# * with sudo privileges *
# Edit the file '/etc/default/docker' and add the following line:
#
# DOCKER_OPTS="--dns 8.8.8.8" 
#
# 2) 
# Then, restart the Docker daemon:
# sudo systemctl restart docker
#
# 3) 
# Build your docker image

#####################################################################
FROM python:3.8-buster

LABEL Maintainer="Nirvash: github@Type-Zero-Nirvash"

WORKDIR home

COPY requirements.txt /home/

RUN pip3 install -r requirements.txt
#####################################################################

#####################################################################
# To run any example, we will use a volume. 
#
# Since many examples produce files for results, graphs, etc
# We want to save to our hostmachine so they can be used after 
# our container has stopped
#
# MAKE SURE YOU ARE ON THE BRANCH THAT CONTAINS THE EXAMPLE YOU
# WANT TO RUN!
#
# Example:
# (To run LineReg.py -> confirm you are on the Linear branch)
# (git checkout Linear)
#
# Linux:
# sudo docker run --rm -v $PWD/Linear:/home nirvash/mumbo python3 /home/LineReg.py --save
#
# Windows:
# docker run --rm -v %cd%/Linear:/home nirvash/mumbo python3 /home/LineReg.py --save
#
# Explanation:
# --rm: will remove the container once it stops
# -v OR --volume: Bind mount a volume 
# $PWD or %cd% will be replaced with the current directory,
# on Linux and Windows respectively
#
#####################################################################
