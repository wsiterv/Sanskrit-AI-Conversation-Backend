#!/usr/bin/env bash
set -ex
apt-get update
apt-get install -y ffmpeg
pip install --upgrade pip setuptools wheel
