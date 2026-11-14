#!/usr/bin/env bash
# pin.sh <command...> -- pin this MPI rank with taskset to its own block of CPUs, taken from the CPUs the
# job is ALLOWED to use (PBS cgroup). Fallback for clusters where Open MPI's own binding fails
# ("hwloc_set_cpubind returned Error"); use together with `mpirun --bind-to none`.
#   PIN_THREADS=<t>  CPUs per rank (OpenMP threads), default 1. Rank r gets allowed[r*t .. r*t+t-1], with the
#                    allowed CPUs ordered by (NUMA node, CPU id) so that consecutive ranks fill one socket first.
r=${OMPI_COMM_WORLD_LOCAL_RANK:-${PMI_LOCAL_RANK:-0}}; t=${PIN_THREADS:-1}
allowed=$(taskset -pc $$ | sed 's/.*: *//')
mapfile -t cpus < <(
  lscpu -p=CPU,NODE 2>/dev/null | grep -v '^#' | awk -F, -v A="$allowed" '
    BEGIN { n = split(A, parts, ","); for (i = 1; i <= n; i++) { split(parts[i], ab, "-"); lo = ab[1]; hi = (ab[2] == "" ? ab[1] : ab[2]); for (c = lo; c <= hi; c++) ok[c] = 1 } }
    ok[$1] { print $2 "," $1 }' | sort -t, -k1,1n -k2,2n | cut -d, -f2)
[ ${#cpus[@]} -eq 0 ] && exec "$@"                       # could not determine: run unpinned
sel=""; for ((i = r * t; i < r * t + t; i++)); do sel="${sel:+$sel,}${cpus[$((i % ${#cpus[@]}))]}"; done
[ -n "${PIN_VERBOSE:-}" ] && echo "rank $r on $(hostname): cpus $sel (allowed: $allowed)" >&2
exec taskset -c "$sel" "$@"
