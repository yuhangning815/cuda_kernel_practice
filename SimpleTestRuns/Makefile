# ---- config ----
NVCC        ?= nvcc
ARCH        ?= sm_80            # override: make ARCH=sm_75 / sm_89 / sm_90
CUTLASS_DIR ?= ../third-party/cutlass

SRC := print_layout.cu
BIN := print_layout
SRC2 := copy.cu
BIN2 := copy

NVCCFLAGS := -std=c++17 -O0 -g -lineinfo
INCLUDES  := -I$(CUTLASS_DIR)/include

# ---- targets ----
.PHONY: all run clean

all: $(BIN)

$(BIN): $(SRC)
	$(NVCC) $(NVCCFLAGS) -arch=$(ARCH) $(INCLUDES) $< -o $@

$(BIN2): $(SRC2)
	$(NVCC) $(NVCCFLAGS) -arch=$(ARCH) $(INCLUDES) $< -o $@


run: $(BIN)
	./$(BIN)

clean:
	rm -f $(BIN)
