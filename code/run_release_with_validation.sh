#!/bin/bash

#with validation layers and debug symbols
RUSTFLAGS=-g cargo run --release --features "validation"
