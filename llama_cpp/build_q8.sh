#!/bin/bash
# Whitrum AI - Build llama.cpp and quantize to Q8
# Founder: Oguzhan (Dr0xy-Drawn)

echo "=================================="
echo "Whitrum AI - llama.cpp Q8 Builder"
echo "Founder: Oguzhan (Dr0xy-Drawn)"
echo "=================================="

# Step 1: Clone llama.cpp
if [ ! -d "llama.cpp" ]; then
    echo "[1/5] Cloning llama.cpp..."
    git clone https://github.com/ggerganov/llama.cpp.git
else
    echo "[1/5] llama.cpp already exists"
fi

# Step 2: Build
echo "[2/5] Building llama.cpp..."
cd llama.cpp
make -j$(nproc)
cd ..

# Step 3: Convert model to GGUF
echo "[3/5] Converting to GGUF (F16)..."
python3 llama.cpp/convert_hf_to_gguf.py \
    ./whitrum-350m-full \
    --outfile ./whitrum-350m-f16.gguf

# Step 4: Quantize to Q8_0
echo "[4/5] Quantizing to Q8_0..."
./llama.cpp/llama-quantize \
    ./whitrum-350m-f16.gguf \
    ./whitrum-350m-q8_0.gguf \
    Q8_0

# Step 5: Verify
echo "[5/5] Verifying..."
ls -lh ./whitrum-350m-q8_0.gguf

echo ""
echo "Done! Model saved to: ./whitrum-350m-q8_0.gguf"
echo ""
echo "Usage:"
echo "  python llama_cpp/run_model.py --model whitrum-350m-q8_0.gguf --interactive"
