#!/bin/sh
# Full sweep. Env: THREADS / RANKS lists. CUDA runs only if ./graph_cuda exists (build with nvcc on a GPU machine).
cd "$(dirname "$0")/.." || exit 1
export OMPI_ALLOW_RUN_AS_ROOT=1 OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1 OMPI_MCA_rmaps_base_oversubscribe=1 OMPI_MCA_mpi_yield_when_idle=1
THREADS=${THREADS:-"1 2 4"}; RANKS=${RANKS:-"1 2 4"}; OUT=results/all_results.csv; CONV=results/convergence.csv
rm -f $OUT $CONV
for spec in rmat18:powerlaw road512:road rmat20:powerlaw road1024:road; do
  g=${spec%%:*}; ty=${spec##*:}
  for T in $THREADS; do ./graph_omp -g data/$g.bin -n $g -y $ty -T $T -r 3 -R 8 -o $OUT -c $CONV; done
  for P in $RANKS;   do mpirun.openmpi -n $P ./graph_mpi -g data/$g.bin -n $g -y $ty -r 3 -R 8 -o $OUT -c $CONV; done
  [ -x ./graph_cuda ] && ./graph_cuda -g data/$g.bin -n $g -y $ty -r 3 -R 8 -o $OUT -c $CONV
done
