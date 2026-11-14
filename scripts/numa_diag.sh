#!/bin/bash
# Print the CPU/NUMA layout of the node and which of its cores this job was given.
# Used to state in the report, as fact, how the cores of a 64-core allocation split over the sockets.
CPUS=$(taskset -cp $$ | sed 's/.*: //')
echo "== job allowed cpus: $CPUS"
echo "== lscpu"; lscpu | grep -E "^(Model name|Socket|Core|Thread|CPU\(s\)|NUMA|Vendor)" 
echo "== numactl -H"; (numactl -H 2>&1 || echo "numactl not available") | head -20
echo "== cpu -> numa node, socket (all cores)"
lscpu -p=CPU,CORE,SOCKET,NODE | grep -v '^#' | tr '\n' ' ' | fold -w 150; echo
echo "== allowed cores per NUMA node / socket"
python3 - "$CPUS" <<'PY'
import sys,subprocess,collections
allowed=set()
for part in sys.argv[1].split(','):
    a,_,b=part.partition('-'); allowed.update(range(int(a),int(b or a)+1))
out=subprocess.check_output(['lscpu','-p=CPU,SOCKET,NODE']).decode()
node=collections.Counter(); sock=collections.Counter(); tot=collections.Counter()
for l in out.splitlines():
    if l.startswith('#'): continue
    c,s,n=map(int,l.split(','))
    tot[n]+=1
    if c in allowed: node[n]+=1; sock[s]+=1
print('allowed cpus:',len(allowed))
for n in sorted(tot): print('NUMA node %d: %d of %d cores allowed'%(n,node[n],tot[n]))
for s in sorted(sock): print('socket %d: %d allowed'%(s,sock[s]))
PY
