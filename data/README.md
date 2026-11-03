# Ku ta vendosësh dataset-in

Shkarko **Environmental Sensor Telemetry Data** nga Kaggle:

https://www.kaggle.com/datasets/garystafford/environmental-sensor-data-132k

(Nevojitet llogari falas Kaggle. Butoni "Download" shkarkon një `.zip`;
brenda tij gjendet një skedar i vetëm CSV.)

Çfarëdo emri ta ketë skedari CSV brenda zip-it (zakonisht
`iot_telemetry_data.csv`), **riemërtoje dhe vendose saktësisht këtu**:

```
data/iot_telemetry_data.csv
```

Domethënë, pas shkarkimit:

```bash
cd data/
unzip ~/Downloads/environmental-sensor-data-132k.zip
mv <emri_i_skedarit_qe_del>.csv iot_telemetry_data.csv
```

## Si ta kontrollosh që e ke vendosur saktë

```bash
python run_experiment.py --data data/iot_telemetry_data.csv --policy adaptive --rounds 5
```

Nëse gjithçka është në rregull, do të shohësh një listë me **3 pajisje**
(ID-të janë adresa tip MAC, p.sh. `b8:27:eb:bf:9d:51`) dhe numrin e
rreshtave/datat e secilës. Nëse del gabim për kolona që mungojnë, hap CSV-në
dhe krahaso headerin me atë të pritur (shih `src/config.py`,
`PRESET_IOT_TELEMETRY`):

```
ts, device, co, humidity, light, lpg, motion, smoke, temp
```

Nëse Kaggle e ka ndryshuar ndërkohë emrin e kolonave, më trego saktësisht
headerin që del dhe e përshtas `src/config.py` në një minutë.
