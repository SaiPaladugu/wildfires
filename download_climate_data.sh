#!/bin/bash
# Download daily climate data from Environment Canada for fire-prone province stations
# Data: 1972-2024, 16 stations across BC, AB, SK, MB, ON, QC, NT, YT
# Each request downloads one year of daily data per station

DATA_DIR="data/climate"
mkdir -p "$DATA_DIR"

# Station ID, start year, end year, name (for filename)
STATIONS=(
    "568,1972,2024,Barkerville_BC"
    "707,1972,2023,Agassiz_BC"
    "2265,1972,2024,Lethbridge_AB"
    "2315,1972,2024,Taber_AB"
    "2925,1972,2024,IndianHead_SK"
    "2973,1972,2023,Muenster_SK"
    "3605,1972,2024,Gretna_MB"
    "3641,1972,2022,Oakbank_MB"
    "4859,1972,2024,Belleville_ON"
    "4715,1972,2024,Windsor_ON"
    "5345,1972,2024,Danville_QC"
    "5440,1972,2024,Richmond_QC"
    "1635,1972,2024,Yohin_NT"
    "1633,1972,2023,CapeParry_NT"
    "1556,1972,2024,HainesJunction_YT"
    "1596,1972,2023,ShinglePoint_YT"
)

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
