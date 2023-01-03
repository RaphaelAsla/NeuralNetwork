#!/usr/bin/sh
echo "Building..."
clang++ main.cpp -o main -std=c++20 -O3
echo "Done!"
