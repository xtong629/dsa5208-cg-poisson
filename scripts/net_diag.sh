#!/usr/bin/env bash
# net_diag.sh <cluster> -- inside a 2-node allocation: network facts + does an alternative MPI stack work across nodes?
set -u
cd "$(dirname "$0")/.."
source scripts/lib.sh
load_config "${1:?usage: net_diag.sh <atlas|vanda>}"
init="$(cfg ENV_INIT)"; [ -n "$init" ] && env_eval "$init"
NODES=($(sort -u "$PBS_NODEFILE")); A=${NODES[0]}; B=${NODES[1]}
echo "nodes: $A $B"
echo "######## network facts (each node)"
for n in $A $B; do
  echo "=== $n"
  ssh -o BatchMode=yes $n 'ip -br addr | grep -v "^lo"; echo "-- ibv devices:"; ibv_devinfo 2>/dev/null | grep -E "hca_id|state|link_layer" | paste - - - | head -8; echo "-- hosts:"; getent hosts $(hostname) $(hostname -f); echo "-- rdma/ib links:"; (rdma link 2>/dev/null || ibstat 2>/dev/null | grep -E "^CA|State|Rate" ) | head -12'
done
echo "######## reachability $A -> $B"
ssh -o BatchMode=yes $A "ping -c 2 -W 2 $B | tail -3; getent hosts $B"
BIP=$(ssh -o BatchMode=yes $B "ip -4 -br addr | awk '\$1==\"ens6f1\" {print \$3}' | cut -d/ -f1")
echo "ens6f1 address of $B: $BIP"; [ -n "$BIP" ] && ssh -o BatchMode=yes $A "ping -c 2 -W 2 $BIP | tail -3"
echo "######## raw TCP between nodes (python sockets, port 40123)"
ssh -o BatchMode=yes $B "timeout 20 python3 -c \"import socket;s=socket.socket();s.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1);s.bind(('',40123));s.listen(1);c,a=s.accept();print('server got',a);c.send(b'ok')\"" &
sleep 3
ssh -o BatchMode=yes $A "timeout 10 python3 -c \"import socket;s=socket.create_connection(('$B',40123),5);print('client got',s.recv(10))\"" || echo "TCP via hostname FAILED"
wait

echo; echo "######## alternative stack: cluster Open MPI 4.1.8 + gcc 13.1 (modules)"
type module >/dev/null 2>&1 || source /etc/profile
module unload nvhpc-hpcx-cuda12/26.1 hpcx >/dev/null 2>&1
module load gcc/13.1.0 openmpi/gcc/64/4.1.8 2>&1 | tail -2
echo "mpirun=$(command -v mpirun)  $(mpirun --version 2>&1 | head -1)"; ompi_info 2>/dev/null | grep -E "plm: tm|ras: tm|MCA (pml|btl)" | head -12
make mpi BUILD=build_b CXX=g++ MPICXX=mpicxx 2>&1 | tail -2
NP=$(wc -l < "$PBS_NODEFILE")
T=60
c() { local label="$1"; shift; echo; echo "#### $label"; timeout -k 5 $T "$@" 2>&1 | grep -v "^\[.*\(UCX\|LOG_CAT\)" | tail -4 | cut -c1-200; echo "rc=${PIPESTATUS[0]}"; }
c "A: no hostfile (does it see the PBS allocation itself?)" mpirun -np $NP --bind-to none hostname
c "B: hostfile, default transport" mpirun --hostfile $PBS_NODEFILE -np $NP --bind-to none build_b/mpi --dim 3 --N 64 --iters 20
c "C: hostfile, ob1+tcp" mpirun --hostfile $PBS_NODEFILE -np $NP --bind-to none --mca pml ob1 --mca btl self,vader,tcp build_b/mpi --dim 3 --N 64 --iters 20
c "D: hostfile, ob1+tcp ens6f1" mpirun --hostfile $PBS_NODEFILE -np $NP --bind-to none --mca pml ob1 --mca btl self,vader,tcp --mca btl_tcp_if_include ens6f1 build_b/mpi --dim 3 --N 64 --iters 20
