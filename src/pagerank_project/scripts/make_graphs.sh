#!/bin/sh
# Power-law (Graph500 RMAT, edge factor 16) vs road-like (2D lattice) at matched vertex counts.
cd "$(dirname "$0")/.." || exit 1
./gen rmat 14 16 1 data/rmat14.bin;  ./gen grid 128  1 data/road128.bin     # small: validation
./gen rmat 18 16 1 data/rmat18.bin;  ./gen grid 512  1 data/road512.bin     # 262k vertices
./gen rmat 20 16 1 data/rmat20.bin;  ./gen grid 1024 1 data/road1024.bin    # 1M vertices
