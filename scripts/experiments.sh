# experiments.sh -- the experiment bodies. Each function runs inside an already-allocated job (or
# locally for the smoke test) and appends one CSV row per run to results/<cluster>/<exp>.csv.
# Rank counts for scaling experiments come from NP_LIST_JOB (colon-separated, set by submit.sh so
# that each PBS job only runs the rank counts that fit its allocation) or from config/cluster.env.
# sourced by run_all.sh after lib.sh and the cluster config.

job_nps() {                # job_nps <default list>   -> space-separated list for this job
  if [ -n "${NP_LIST_JOB:-}" ]; then echo "$NP_LIST_JOB" | tr ':' ' '; else echo "$1"; fi
}

# ---- 1. serial: runtime vs problem size (also provides T_s for speedup/efficiency) -----------
exp_serial() {
  init_run serial cpu; need_bin serial
  local dim N rep sizes
  for dim in 2 3; do
    if [ "$dim" = 2 ]; then sizes="$SERIAL_FIXED_2D"; else sizes="$SERIAL_FIXED_3D"; fi
    for N in $sizes; do                       # fixed iteration count: time/iteration vs size, and T_s
      for rep in $(seq 1 "$REPS"); do
        OMP_NUM_THREADS=1 run_logged "$REPO_ROOT/build/serial" --dim $dim --N "$N" --iters "$ITERS" \
          --csv "$CSV" --tag serial_fixed --rep "$rep"
      done
    done
    if [ "$dim" = 2 ]; then sizes="$SERIAL_CONV_2D"; else sizes="$SERIAL_CONV_3D"; fi
    for N in $sizes; do                       # run to tolerance: total time to solution vs size
      for rep in $(seq 1 "$REPS_CONV"); do
        OMP_NUM_THREADS=1 run_logged "$REPO_ROOT/build/serial" --dim $dim --N "$N" --tol "$TOL" \
          --csv "$CSV" --tag serial_conv --rep "$rep"
      done
    done
  done
}

# ---- 2. MPI strong scaling (fixed N) + Allreduce latency -------------------------------------
exp_strong() {
  init_run strong cpu; need_bin mpi
  local dim N np rep sizes
  for dim in 2 3; do
    if [ "$dim" = 2 ]; then sizes="$STRONG_N_2D"; else sizes="$STRONG_N_3D"; fi
    for N in $sizes; do
      for np in $(job_nps "$NP_LIST $NP_EXTRA"); do
        pts_ok $dim "$N" "$np" "$STRONG_MAX_PTS_PER_RANK" || continue
        for rep in $(seq 1 "$REPS"); do
          run_mpi "$np" mpi --dim $dim --N "$N" --iters "$ITERS" --decomp block --halo nonblocking \
            --overlap 1 --csv "$CSV" --tag strong --rep "$rep"
        done
      done
    done
  done
  for np in $(job_nps "$NP_LIST $NP_EXTRA"); do   # pure MPI_Allreduce latency (1 double) at this rank count
    for rep in $(seq 1 "$REPS"); do
      run_mpi "$np" mpi --bench-allreduce 5000 --csv "$RESDIR/allreduce.csv" --tag allreduce --rep "$rep"
    done
  done
}

# ---- 3. MPI weak scaling (fixed points per rank) ---------------------------------------------
exp_weak() {
  init_run weak cpu; need_bin mpi
  local dim base np rep N bases
  for dim in 2 3; do
    if [ "$dim" = 2 ]; then bases="$WEAK_BASE_2D"; else bases="$WEAK_BASE_3D"; fi
    for base in $bases; do
      for np in $(job_nps "$NP_LIST $NP_EXTRA"); do
        N="$(weak_N "$base" "$np" $dim)"
        pts_ok $dim "$N" "$np" || continue
        for rep in $(seq 1 "$REPS"); do
          run_mpi "$np" mpi --dim $dim --N "$N" --iters "$ITERS" --decomp block --halo nonblocking \
            --overlap 1 --csv "$CSV" --tag "weak_base$base" --rep "$rep"
        done
      done
    done
  done
}

# ---- 4. decomposition (slab vs block) x halo variant (blocking / non-blocking / overlap) ------
exp_decomp_halo() {
  init_run decomp_halo cpu; need_bin mpi
  local dim N np rep decomp v halo ov
  for dim in 2 3; do
    if [ "$dim" = 2 ]; then N="$DH_N_2D"; else N="$DH_N_3D"; fi
    for np in $(job_nps "$DH_NP_LIST"); do
      pts_ok $dim "$N" "$np" || continue
      for decomp in slab block; do
        [ "$decomp" = slab ] && [ "$N" -lt "$np" ] && continue
        for v in blocking:0 nonblocking:0 nonblocking:1; do
          halo="${v%%:*}"; ov="${v##*:}"
          for rep in $(seq 1 "$REPS"); do
            run_mpi "$np" mpi --dim $dim --N "$N" --iters "$ITERS" --decomp "$decomp" --halo "$halo" \
              --overlap "$ov" --csv "$CSV" --tag decomp_halo --rep "$rep"
          done
        done
      done
    done
  done
}

# ---- 5. hybrid MPI+OpenMP at fixed total cores (ranks x threads = total) ---------------------
exp_hybrid() {
  init_run hybrid cpu; need_bin hybrid mpi
  local dim N total th np rep cpn
  cpn="$(cfg CORES_PER_NODE)"
  for dim in 2 3; do
    if [ "$dim" = 2 ]; then N="$HYB_N_2D"; else N="$HYB_N_3D"; fi
    for total in $(job_nps "$HYB_TOTAL_CORES"); do
      for th in $HYB_THREADS; do
        [ $(( total % th )) -eq 0 ] || continue
        [ "$th" -le "$cpn" ] || continue           # a rank's threads must fit in one node
        np=$(( total / th ))
        pts_ok $dim "$N" "$np" || continue
        for rep in $(seq 1 "$REPS"); do
          run_hybrid "$np" "$th" hybrid --dim $dim --N "$N" --iters "$ITERS" --decomp block \
            --halo nonblocking --overlap 1 --csv "$CSV" --tag "hybrid_total$total" --rep "$rep"
        done
      done
      for rep in $(seq 1 "$REPS"); do               # reference: the pure-MPI binary on the same cores
        run_mpi "$total" mpi --dim $dim --N "$N" --iters "$ITERS" --decomp block --halo nonblocking \
          --overlap 1 --csv "$CSV" --tag "hybrid_total${total}_pure_mpi" --rep "$rep"
      done
    done
  done
}

# ---- 6a. GPU: problem-size sweep on 1 GPU, against one CPU node (same allocation) ------------
exp_gpu_sweep() {
  init_run gpu_sweep gpu; need_bin cuda mpi
  local dim N rep sizes cpus
  cpus="$(cfg GPU_NODE_CPUS)"
  echo "GPU self-test before spending time on runs ..."
  OMP_NUM_THREADS=1 $MPIRUN -np 1 "$REPO_ROOT/build/cuda" --selftest || { echo "selftest failed; aborting" >&2; exit 3; }
  for dim in 2 3; do
    if [ "$dim" = 2 ]; then sizes="$GPU_SIZES_2D"; else sizes="$GPU_SIZES_3D"; fi
    for N in $sizes; do
      for rep in $(seq 1 "$REPS"); do
        run_logged $MPIRUN -np 1 "$REPO_ROOT/build/cuda" --dim $dim --N "$N" --iters "$ITERS" \
          --halo nonblocking --overlap 1 --csv "$CSV" --tag gpu_1gpu --rep "$rep"
      done
      if pts_ok $dim "$N" "$cpus"; then           # one CPU node, pure MPI, same sizes
        for rep in $(seq 1 "$REPS"); do
          run_mpi "$cpus" mpi --dim $dim --N "$N" --iters "$ITERS" --decomp block --halo nonblocking \
            --overlap 1 --csv "$CSV" --tag cpu_node --rep "$rep"
        done
      fi
    done
  done
}

# ---- 6b. multi-GPU strong scaling on one node + CUDA-aware vs host-staged vs overlap ---------
exp_gpu_multi() {
  init_run gpu_multi gpu; need_bin cuda
  local dim N np rep v halo ov st gmax nps
  gmax="$(cfg GPUS_PER_NODE)"
  nps="$(awk -v g="$gmax" 'BEGIN { n = 1; while (n <= g) { printf "%d ", n; last = n; n *= 2 } if (last != g) printf "%d", g }')"
  [ -n "${NP_LIST_JOB:-}" ] && nps="$(echo "$NP_LIST_JOB" | tr ':' ' ')"
  OMP_NUM_THREADS=1 $MPIRUN -np 1 "$REPO_ROOT/build/cuda" --selftest || { echo "selftest failed; aborting" >&2; exit 3; }
  for dim in 2 3; do
    if [ "$dim" = 2 ]; then N="$GPU_MULTI_N_2D"; else N="$GPU_MULTI_N_3D"; fi
    for np in $nps; do
      for v in nonblocking:1:0 nonblocking:0:0 blocking:0:0 nonblocking:1:1; do   # halo:overlap:staging
        halo="$(echo "$v" | cut -d: -f1)"; ov="$(echo "$v" | cut -d: -f2)"; st="$(echo "$v" | cut -d: -f3)"
        [ "$np" -eq 1 ] && [ "$v" != "nonblocking:1:0" ] && continue   # no neighbours: variants coincide
        for rep in $(seq 1 "$REPS"); do
          run_logged $MPIRUN -np "$np" "$REPO_ROOT/build/cuda" --dim $dim --N "$N" --iters "$ITERS" \
            --decomp block --halo "$halo" --overlap "$ov" --staging "$st" --csv "$CSV" \
            --tag gpu_multi --rep "$rep"
        done
      done
    done
  done
}
