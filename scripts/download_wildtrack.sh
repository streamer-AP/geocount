#!/bin/bash
# Persistent Wildtrack download script - keeps retrying until complete
# EPFL server is heavily throttled (~20KB/s) and drops connections every 2-3 min
# wget -c handles resume; this script loops until the file is fully downloaded

URL="https://documents.epfl.ch/groups/c/cv/cvlab-unit/www/data/Wildtrack/Wildtrack_dataset_full.zip"
OUTPUT="/root/geocount/data/Wildtrack_dataset_full.zip"
EXPECTED_SIZE=6807496358

while true; do
    CURRENT_SIZE=$(stat -c%s "$OUTPUT" 2>/dev/null || echo 0)
    if [ "$CURRENT_SIZE" -ge "$EXPECTED_SIZE" ]; then
        echo "Download complete! Size: $CURRENT_SIZE bytes"
        break
    fi
    echo "[$(date)] Resuming download... Current: ${CURRENT_SIZE} / ${EXPECTED_SIZE} bytes ($(( CURRENT_SIZE * 100 / EXPECTED_SIZE ))%)"
    wget -c --timeout=30 --tries=3 --wait=2 --retry-connrefused \
         --no-check-certificate \
         -O "$OUTPUT" "$URL" 2>&1 | tail -1
    echo "[$(date)] Connection dropped, retrying in 3s..."
    sleep 3
done

echo "Verifying ZIP integrity..."
python3 -c "
import zipfile
z = zipfile.ZipFile('$OUTPUT')
print(f'ZIP OK: {len(z.namelist())} entries')
z.close()
" 2>&1 || echo "ZIP verification failed - file may still be truncated"
