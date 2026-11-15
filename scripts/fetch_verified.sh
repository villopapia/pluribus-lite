#!/usr/bin/env bash
# Copy files from a Windows host over ssh/scp and verify them by SHA-256 (hashes computed on the
# remote side with Get-FileHash, checked locally with sha256sum). Read-only on the remote.
#
#   scripts/fetch_verified.sh HOST REMOTE_DIR LOCAL_DIR PATTERN...
#   e.g. scripts/fetch_verified.sh user@desktop-g1k8ugg C:/Users/User/pluribus-lite/data/runs/pilot_fit11_long \
#          data/runs/pilot_fit11_long_final 'ckpt_*.bin' schedule.txt
#
# PATTERN is a file name or a wildcard inside REMOTE_DIR. Leaves LOCAL_DIR/remote.sha256 (the
# remote hashes) and exits non-zero unless every remote file arrived with the same hash.
set -euo pipefail
host=$1; rdir=${2//\\//}; ldir=$3; shift 3
mkdir -p "$ldir"
inc=$(printf "'%s'," "$@"); inc=${inc%,}
ps="\$ProgressPreference = 'SilentlyContinue'; Get-ChildItem -Path '${rdir//\//\\}\\*' -File -Include $inc | Sort-Object Name | ForEach-Object { (Get-FileHash -Algorithm SHA256 -LiteralPath \$_.FullName).Hash.ToLower() + '  ' + \$_.Name }"
enc=$(printf '%s' "$ps" | iconv -f utf-8 -t utf-16le | base64 -w0)
ssh "$host" "powershell -NoProfile -NonInteractive -EncodedCommand $enc" | tr -d '\r' > "$ldir/remote.sha256"
n=$(wc -l < "$ldir/remote.sha256")
[ "$n" -gt 0 ] || { echo "no remote files match: $*" >&2; exit 1; }
echo "$n remote files; copying"
srcs=(); for p in "$@"; do srcs+=("$host:$rdir/$p"); done
scp -q "${srcs[@]}" "$ldir/"
(cd "$ldir" && sha256sum --quiet -c remote.sha256) && echo "OK: $n files verified (SHA-256) in $ldir"
