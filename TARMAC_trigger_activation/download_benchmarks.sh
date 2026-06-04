#!/bin/bash

# Create benchmarks directory
mkdir -p benchmarks
cd benchmarks

echo "======================================"
echo "Downloading OpenRISC mor1kx..."
echo "======================================"
git clone https://github.com/openrisc/mor1kx.git mor1kx
echo "mor1kx downloaded successfully."

echo "======================================"
echo "Downloading AES-128 (from bomberman)..."
echo "======================================"
mkdir -p AES_128
cd AES_128
# Use sparse checkout to only get the third_party/aes_128 folder
git clone --no-checkout https://github.com/timothytrippel/bomberman.git temp_bomberman
cd temp_bomberman
git sparse-checkout init --cone
git sparse-checkout set third_party/aes_128
git checkout master
mv third_party/aes_128/* ../
cd ..
rm -rf temp_bomberman
echo "AES-128 downloaded successfully."

echo "======================================"
echo "Downloading GPS (from CEP)..."
echo "======================================"
mkdir -p GPS
cd GPS
# Use sparse checkout to only get the gps folder
git clone --no-checkout https://github.com/CommonEvaluationPlatform/CEP.git temp_cep
cd temp_cep
git sparse-checkout init --cone
git sparse-checkout set generators/mitll-blocks/src/main/resources/vsrc/gps
git checkout master
mv generators/mitll-blocks/src/main/resources/vsrc/gps/* ../
cd ..
rm -rf temp_cep
echo "GPS downloaded successfully."

echo "======================================"
echo "Benchmark Collection Complete!"
echo "======================================"
echo "NOTE: Trust-Hub TRIT-TC benchmarks cannot be downloaded via Git."
echo "Please visit https://www.trust-hub.org/, navigate to the Benchmarks section,"
echo "and manually download the TRIT-TC (ISCAS'85 and ISCAS'89) datasets."
echo "Extract them into a new folder here: TARMAC_trigger_activation/benchmarks/TRIT-TC/"
