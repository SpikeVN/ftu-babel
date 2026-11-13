#!/usr/bin/env bash
# ==============================================================================
# sync_to_xavier.sh
# 
# 1. Prepares /mnt/ssd/babel on xavier (copies existing config/compose from eMMC).
# 2. Vacuums stash.db into transfer.db with sqlite-vec extension.
# 3. Compresses transfer.db using zstd (-T0).
# 4. Rsyncs transfer.db.zst to triph@xavier:/mnt/ssd/babel/.
# 5. Unzstds on xavier, updates stash.db, and deletes remote .zst.
# 6. Deletes local .zst compressed file.
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

DB_SOURCE="${1:-stash.db}"
TRANSFER_DB="transfer.db"
TRANSFER_ZST="transfer.db.zst"
REMOTE_HOST="triph@xavier"
REMOTE_TARGET_DIR="/mnt/ssd/babel"
REMOTE_OLD_DIR="/home/triph/projects/babel"

echo "==================================================================="
echo "  Babel DB Sync to Xavier (/mnt/ssd/babel)"
echo "==================================================================="

# 0. Check prerequisites
if [ ! -f "$DB_SOURCE" ]; then
    echo "Error: Database source '$DB_SOURCE' not found in $SCRIPT_DIR!" >&2
    exit 1
fi

command -v zstd >/dev/null 2>&1 || { echo "Error: 'zstd' command not found locally." >&2; exit 1; }
command -v rsync >/dev/null 2>&1 || { echo "Error: 'rsync' command not found locally." >&2; exit 1; }
command -v ssh >/dev/null 2>&1 || { echo "Error: 'ssh' command not found locally." >&2; exit 1; }

echo "[0/5] Verifying SSH connection to $REMOTE_HOST..."
ssh -o BatchMode=yes -o ConnectTimeout=8 "$REMOTE_HOST" "echo 'SSH connection OK'"

# 1. Prepare remote /mnt/ssd/babel and copy old setup from eMMC if needed
echo "[1/5] Preparing $REMOTE_TARGET_DIR on xavier (NVMe SSD)..."
ssh -o BatchMode=yes "$REMOTE_HOST" bash -s <<EOF
set -euo pipefail
mkdir -p "$REMOTE_TARGET_DIR"

# If compose.yml or .env doesn't exist yet on /mnt/ssd/babel, copy from old eMMC directory
if [ -d "$REMOTE_OLD_DIR" ] && [ ! -f "$REMOTE_TARGET_DIR/compose.yml" ]; then
    echo "Copying existing configuration from $REMOTE_OLD_DIR to $REMOTE_TARGET_DIR..."
    cp -a "$REMOTE_OLD_DIR/.env" "$REMOTE_TARGET_DIR/.env" 2>/dev/null || true
    cp -a "$REMOTE_OLD_DIR/compose.yml" "$REMOTE_TARGET_DIR/compose.yml" 2>/dev/null || true
    cp -a "$REMOTE_OLD_DIR/searxng" "$REMOTE_TARGET_DIR/searxng" 2>/dev/null || true
fi
EOF

# 2. Vacuum into transfer.db
echo "[2/5] Vacuuming $DB_SOURCE into $TRANSFER_DB..."
rm -f "$TRANSFER_DB" "$TRANSFER_ZST"

python3 - <<EOF
import sqlite3, sys, time
try:
    import sqlite_vec
except ImportError:
    sqlite_vec = None

t0 = time.time()
conn = sqlite3.connect("$DB_SOURCE", timeout=120.0)
if sqlite_vec is not None:
    try:
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.enable_load_extension(False)
    except Exception as e:
        print(f"Notice: sqlite_vec load warning: {e}", file=sys.stderr)

conn.execute("VACUUM INTO '$TRANSFER_DB'")
conn.close()
print(f"VACUUM completed in {time.time() - t0:.2f}s.")
EOF

# 3. Zstandard compression
echo "[3/5] Compressing $TRANSFER_DB with zstd (multi-threaded)..."
zstd -T0 -f "$TRANSFER_DB" -o "$TRANSFER_ZST"
ls -lh "$TRANSFER_ZST"

# 4. Rsync to xavier
echo "[4/5] Rsyncing $TRANSFER_ZST to $REMOTE_HOST:$REMOTE_TARGET_DIR/..."
rsync -avP --progress "$TRANSFER_ZST" "$REMOTE_HOST:$REMOTE_TARGET_DIR/"

# 5. Uncompress on xavier & update stash.db
echo "[5/5] Extracting on xavier and updating $REMOTE_TARGET_DIR/stash.db..."
ssh -o BatchMode=yes "$REMOTE_HOST" bash -s <<EOF
set -euo pipefail
cd "$REMOTE_TARGET_DIR"

echo "Extracting $TRANSFER_ZST on xavier..."
unzstd -f --rm "$TRANSFER_ZST"

echo "Updating stash.db with fresh vacuumed database..."
cp -f "$TRANSFER_DB" stash.db
chmod 666 stash.db
rm -f "$TRANSFER_DB"

echo "Remote database status on /mnt/ssd/babel:"
ls -lh stash.db
EOF

# 6. Delete local compressed file
echo "Cleaning up local compressed file $TRANSFER_ZST..."
rm -f "$TRANSFER_ZST"

echo "==================================================================="
echo "  Sync to Xavier complete!"
echo "  Files live at: $REMOTE_HOST:$REMOTE_TARGET_DIR/"
echo "==================================================================="
