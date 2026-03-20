#!/bin/bash
# Download daily climate data from Environment Canada for fire-prone province stations
# Data: 1972-2024, 16 stations across BC, AB, SK, MB, ON, QC, NT, YT
# Each request downloads one year of daily data per station

DATA_DIR="data/climate"
STATIONS_CSV="data/stations_1972_2024.csv"
mkdir -p "$DATA_DIR"

# Read station ID, start year, end year, name from CSV (skip header)
mapfile -t STATIONS < <(tail -n +2 "$STATIONS_CSV")

BASE_URL="https://climate.weather.gc.ca/climate_data/bulk_data_e.html"
TOTAL=0
DOWNLOADED=0

# Count total downloads
for entry in "${STATIONS[@]}"; do
    IFS=',' read -r sid start end name <<< "$entry"
    for year in $(seq "$start" "$end"); do
        TOTAL=$((TOTAL + 1))
    done
done

echo "Downloading daily climate data: $TOTAL files from ${#STATIONS[@]} stations"
echo "Output directory: $DATA_DIR"
echo "---"

for entry in "${STATIONS[@]}"; do
    IFS=',' read -r sid start end name <<< "$entry"
    echo "Station: $name (ID=$sid), years $start-$end"

    for year in $(seq "$start" "$end"); do
        OUTFILE="$DATA_DIR/climate_daily_${name}_${year}.csv"

        if [ -f "$OUTFILE" ] && [ -s "$OUTFILE" ]; then
            DOWNLOADED=$((DOWNLOADED + 1))
            continue  # Skip already downloaded
        fi

        curl -s -L -o "$OUTFILE" \
            "${BASE_URL}?format=csv&stationID=${sid}&Year=${year}&Month=1&Day=1&timeframe=2&submit=Download+Data"

        DOWNLOADED=$((DOWNLOADED + 1))

        # Brief pause to be polite to the server
        if [ $((DOWNLOADED % 50)) -eq 0 ]; then
            echo "  Progress: $DOWNLOADED/$TOTAL files"
            sleep 1
        fi
    done
    echo "  Done: $name"
done

echo "---"
echo "Download complete: $DOWNLOADED/$TOTAL files in $DATA_DIR"
ls -lh "$DATA_DIR" | tail -5
echo "Total files: $(ls -1 "$DATA_DIR" | wc -l)"
echo "Total size: $(du -sh "$DATA_DIR" | cut -f1)"
