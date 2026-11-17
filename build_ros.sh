#!/bin/bash

# FREE_MEM_THRESH=$((8 * 1024 * 1024 * 1024))
# BYTES_PER_JOB=$((3 * 1024 * 1024 * 1024))

# if [[ "$(uname)" == "Linux" ]]; then
#     AVAIL_MEM=$(awk '/MemAvailable/ { print $2 * 1024 }' /proc/meminfo)
# elif [[ "$(uname)" == "Darwin" ]]; then
#     PAGE_SIZE=$(sysctl -n hw.pagesize)
#     FREE_PAGES=$(vm_stat | awk '/Pages free/ { gsub("\\.", "", $3); print $3 }')
#     AVAIL_MEM=$((FREE_PAGES * PAGE_SIZE))
# fi

# if [ "$AVAIL_MEM" -le "$FREE_MEM_THRESH" ]; then
#     JOBS=$((AVAIL_MEM / BYTES_PER_JOB))
#     THREADS=$(getconf _NPROCESSORS_ONLN)
#     if (( JOBS < 1 )); then JOBS=1; fi
#     if (( JOBS > THREADS )); then JOBS=$THREADS; fi

#     echo "USING $JOBS THREADS FOR BUILD"
#     export MAKEFLAGS="-j $JOBS"
# fi


START_TIME=$(date +%s.%N)

colcon build \
    --symlink-install \
    --executor parallel \
    --event-handlers console_direct+ \
    --cmake-args \
        -Wno-dev \
        -DCMAKE_EXPORT_COMPILE_COMMANDS:BOOL=ON
        # -DCMAKE_CXX_FLAGS="-march=native" \
        # -DCMAKE_C_FLAGS="-march=native" \

END_TIME=$(date +%s.%N)
DELTA_TIME=$(echo "$END_TIME - $START_TIME" | bc)

echo ">> Build finished in $DELTA_TIME seconds."
