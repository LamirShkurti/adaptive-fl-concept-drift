# Federated Adaptive Forecasting for Heterogeneous IoT Sensor Networks Under Concept Drift

Kodi shoqërues (companion code) i punimit me të njëjtin titull, Lamir Shkurti
et al., 2026 (nën review).

Sistem simulimi (jo-Docker, jo-Flask -- një proces i vetëm Python) për
Federated Learning ku **çdo klient vendos vetë**, çdo raund, bazuar në
gjendjen e tij reale/simuluar: a trajnon fare, sa epochs lokale, sa shpesh
dërgon update, a e dërgon update-in e plotë apo të kompresuar, a e mban
modelin lokal aktiv, dhe a kalon në një variant model më të vogël.

## 1. Çfarë është ky projekt

Është një sistem i ri, i ndërtuar nga e para, që:

- **riperdor metodologjinë e provuar** të feature engineering (lag/rolling/
  ciklike, split kronologjik, skill-score kundrejt persistence) nga punimi i
  parashikimit të temperaturës;
- **kalibron heterogjenitetin e pajisjeve simuluese**
- **shton nga e para** 7 dimensione që s'ekzistonin më parë: bateri/energji,
  RAM si sinjal, kompresim update-i, aktivizim/çaktivizim model, kalim në
  model më të vogël, shpërndarje jo-IID e të dhënave, dhe concept drift.

## 2. Instalimi

```bash
pip install -r requirements.txt
```

## 3. Dataset-i -- SHIH `data/README.md`

Përmbledhje: shkarko **Environmental Sensor Telemetry Data** nga Kaggle
(`garystafford/environmental-sensor-data-132k`) dhe vendose te
`data/iot_telemetry_data.csv`. Pse ky dataset specifik:

- **3 pajisje reale** (Raspberry Pi + DHT22/MQ-sensor array), identifikuar me
  adresë MAC -- bëhen klientët FL natyrshëm jo-IID (vendndodhje/kushte fizike
  të ndryshme), pa nevojë për ndarje artificiale të një dataset-i të vetëm.
- Target: `temp` (temperaturë) -- konsistent me fokusin e punimit të
  mëparshëm.
- I lehtë (CSV i vetëm, pa autentikim API).

## 4. Xhirimi i një eksperimenti

```bash
# Baseline jo-adaptive (si UseCase1_NOAdaptive): trajno gjithmonë, dërgo gjithmonë të plotë.
python run_experiment.py --data data/iot_telemetry_data.csv --policy non_adaptive --rounds 80

# Politika adaptive (kontributi i ri, 6 dimensionet):
python run_experiment.py --data data/iot_telemetry_data.csv --policy adaptive --rounds 80

# Me skenar concept-drift mbi një pajisje specifike (shih §6):
python run_experiment.py --data data/iot_telemetry_data.csv --policy adaptive --rounds 80 \
    --drift "b8:27:eb,gradual_sensor_bias,40,3.0"
```

Çdo xhirim shkruan `outputs/<run-name>.csv` (log për-raund-për-klient) dhe
`outputs/<run-name>_config.json` (parametrat e plotë -- për riprodhueshmëri).

### Krahasimi dhe figurat

```bash
python analyze_results.py \
    --runs outputs/adaptive_80r.csv outputs/non_adaptive_80r.csv \
    --labels Adaptive NonAdaptive \
    --output-dir outputs/figures
```

Prodhon: `summary_table.csv` (RMSE mesatare, bajte totale, % raunde trajnim/
dërgim/kompresim/lite/çaktivizim), `rmse_over_rounds.png`,
`bytes_cumulative.png`, `decisions_<label>.png`.

### Krahasimi i drift-it (figura e Seksionit 4.2 te punimit)

`analyze_results.py` mesatarizon RMSE-në mbi TË TRE klientët -- e dobishme
për baseline-in (§4.1), por e papërshtatshme për drift, sepse vetëm 1 nga 3
klientët ka drift (sinjali "zhytet" te mesatarja). Për këtë përdoret një
skript i veçantë, `plot_drift_comparison.py`, që ndjek vetëm klientin e
targetuar:

```bash
python plot_drift_comparison.py \
    --abrupt-nonadaptive outputs/nonadaptive_80r_drift_abrupt.csv \
    --abrupt-adaptive    outputs/adaptive_80r_drift_abrupt.csv \
    --gradual-nonadaptive outputs/nonadaptive_80r_drift_gradual.csv \
    --gradual-adaptive    outputs/adaptive_80r_drift_gradual.csv \
    --output-dir outputs/figures
```

(`--target`/`--start-round`/`--ramp-rounds` kanë tashmë si default pikërisht
vlerat e komandës së drift-it në §4 më sipër -- lini pa i specifikuar nëse
s'i keni ndryshuar.) Prodhon `drift_rmse_comparison.png` (figura 2-panele, SAKTËSISHT
siç është në punim, Figura 4) dhe `drift_phase_summary.csv` (RMSE mesatare
pre-drift/gjatë/pas, numrat e Tabelës 4). Printon edhe një verifikim
("diferenca max. e RMSE për raundet < start\_round") që duhet të dalë 0 --
nëse del ndryshe nga 0, diçka ka ndryshuar mes xhirimeve (p.sh. `--seed`).

## 5. Arkitektura (9 sinjale -> 6 vendime)

Shih `src/controller.py` (dokumentim i plotë në docstring). Përmbledhje:

**Sinjalet** (`InputSignals`): bateri %, CPU %, RAM e lirë, bandwidth, latencë,
sasia e të dhënave të reja, shmangia nga shpërndarja historike lokale, trendi
i RMSE-së së fundit, raporti i zbulimit të drift-it (EWMA/baseline).

**Vendimet** (`Decision`): trajno/kapërce, epochs lokale, dërgo/mbaj
update-in, i plotë/i kompresuar, model aktiv/jo-aktiv, variant i plotë/lite.

Çdo prag konfigurohet te `src/config.py::ControllerThresholds` -- ndryshoji
lirisht dhe rixhiro për analizë ndjeshmërie (sensitivity analysis), e cila
shpesh pritet nga recensentët për një kontribut "rule-based controller".

## 6. Concept drift (shih `src/drift.py`)

Dataset-i Kaggle mbulon vetëm ~1 javë -- s'ka drift stinor afatgjatë si në
Tetovo. Drift-i **injektohet shprehimisht** (duhet deklaruar kështu në
punim, jo si "zbulim organik"):

- `gradual_sensor_bias`: bias aditiv që rritet ngadalë deri në `magnitude`
  brenda `ramp_rounds`, i aplikuar njëlloj te target-i DHE features e
  lidhura me `temp` (analoge me ngrohjen diellore të node4/UB).
  **Shënim metodologjik i rëndësishëm**: pasi bias-i stabilizohet (mbaron
  ramp-i), një model autoregresiv linear e "përthith" pjesërisht automatikisht
  (i njëjti offset del në lag-features DHE target) -- rritja e gabimit
  REALE, e matshme, ndodh **gjatë vetë ramp-it**, jo domosdoshmërisht gjatë
  gjithë periudhës pasuese. Kjo vetë është një gjetje e vlefshme për punim
  (dallon "drift transitor" nga "drift i qëndrueshëm").
- `abrupt_relationship_shift`: në një moment, shkallëzon marrëdhënien
  humidity->temp -- drift "i papritur" klasik (taksonomia Gama et al. 2014),
  efekti i të cilit **MBETET** (nuk përthithet vetvetiu), i përshtatshëm për
  të treguar rikuperim afatgjatë nga kontrolluesi adaptiv.

Format argumenti CLI: `--drift "<pjesë_ID_pajisjeje>,<lloji>,<raundi_fillimit>,<madhësia>"`
(presje -- jo ":" -- sepse ID-të e pajisjeve Kaggle vetë përmbajnë ":").

### 6.1 FIKS (tetor 2026): `start_round`/`ramp_rounds` ishin në njësi të gabuar

**Nëse keni xhiruar ndonjë eksperiment me `--drift` PARA kësaj date, rixhironi
atë -- rezultatet e vjetra NUK janë të vlefshme.** Eksperimentet PA `--drift`
(baseline-i adaptive/non_adaptive, ablacionet) NUK preken fare nga ky fiks.

Bug-u: `apply_drift_to_frame` (te `src/drift.py`) merrte `live_df.index` (çdo
RRESHT RAW, para feature engineering) dhe e trajtonte çdo rresht si një
"raund" të vetëm (`enumerate()`). Në të dhënat reale (`resample_freq=5min`,
tipikisht ~6-8 rreshta raw/raund me `--rounds 80`), kjo bënte që
`start_round=40` (synuar si RAUNDI 40) të aplikohej efektivisht te **rreshti
raw #40**, pra rreth **raundi ~5-6** -- ~7x më shpejt se sa duhej -- dhe
`ramp_rounds=30` të përfundonte brenda ~4-5 raundeve, jo 30. U zbulua duke
analizuar një batch eksperimentesh reale (RMSE e klientit të targetuar ishte
tashmë e ngritur te raundi 6, shumë përpara `start_round=40` të konfiguruar).
U fiksua duke e llogaritur numrin e saktë të raundit për çdo rresht raw
(`_chronological_round_slices` mbi `len(live_df)`, e njëjta logjikë e
përdorur më vonë për kufijtë përfundimtarë të raundeve) PARA se t'ia kalojmë
`apply_drift_to_frame`, në vend të `live_df.index`.

Pse nuk u kap nga testet ekzistuese: fixture-i i testeve (`tests/fixtures/`)
+ `n_live_rounds` i vogël i përdorur atje bënin që raport rreshta/raund të
mos ishte realisht i madh në mënyrë që të dallohej qartë, DHE testi ekzistues
(`test_drift_injection_raises_error_on_targeted_client`) kontrollon vetëm "a u
rrit gabimi MANDEJ", jo "a u rrit TOCKA kur duhej" -- kalon qoftë me qoftë
pa bug. Testi i ri `test_drift_never_touches_rounds_before_start_round` (te
`tests/test_pipeline.py`) kontrollon pikërisht këtë: me një raport
rreshta/raund >> 1, çdo raund PARA `start_round` duhet të jetë **identik**
mes skenarit "none" dhe skenarit me drift -- dhe DËSHTON në kodin e vjetër
(e verifikova drejtpërdrejt para se ta dërgoj ky fiks), kalon me kodin e ri.

## 7. Modeli i energjisë/baterisë (shih `src/device_state.py`)

Bateria **evoluon realisht** gjatë simulimit (nuk është numër i rastësishëm
çdo raund): zbret sipas rrymës së tërhequr (idle kundrejt training, plus
kosto shtesë për dërgim) dhe kohëzgjatjes së raundit. Kapacitetet/rrymat janë
**supozime inxhinierike të deklaruara** (jo matje -- `UseCase1` nuk mati
baterie reale), të ankoruara te numra tipikë të publikuar (Raspberry Pi
4/5 power draw, power-bank 10-20k mAh) dhe te profili real Arduino MKR1000
nga punimi i mëparshëm (Seksioni 4.5: ~204-259 cikle, 4.3-5.4 µs/parashikim
@ 48MHz). Deklaroje këtë qartë në punim -- mos e paraqit si matje fizike.

## 8. Testet

```bash
python tests/test_pipeline.py
# ose, nëse ke pytest:
python -m pytest tests/ -v
```

9 teste rregresioni: ngarkimi gjen 3 pajisjet, politika adaptive xhiron pa
NaN/Inf, baseline-i jo-adaptiv trajnon/dërgon gjithmonë siç pritet, drift-i
rrit gabimin gjatë ramp-it, drift-i NUK prek raunde para `start_round` (shih
§6.1 FIKS), `prepare_clients()` refuzon `--rounds` shumë të lartë për
rreshta "live" të pakët (shih §9.1), adaptive dërgon <= bajte se
non_adaptive, dhe 2 teste ablacioni (bosht 3 "send-hold", bosht 4
"compression"). Përdorin një fixture TË VOGËL sintetik
(`tests/fixtures/tiny_iot_telemetry.csv`, VETËM për CI -- jo për rezultate
punimi).

## 9. Çfarë duhet VETË (eksperimentet reale)

1. Vendos `data/iot_telemetry_data.csv` (shih §3/`data/README.md`).
2. Xhiro `non_adaptive` dhe `adaptive` (dhe variante drift) me `--rounds`
   mjaftueshëm (sugjeroj 60-120, provo disa vlera -- varet nga sa rreshta ka
   dataset-i pas resample; shih §9.1 PARA se te shkosh mbi ~95 raunde).

### 9.1 KUJDES: `--rounds` i lartë + `--resample-freq` parazgjedhje = ZERO trajnim (gjetur/fiksuar tetor 2026)

Numri total i rreshtave "live" (25% e dataset-it, test\_frac) është **FIKS**
për një `--resample-freq` të dhënë -- rritja e `--rounds` thjesht e ndan të
njëjtën sasi rreshtash në më shumë, më të vegjël copëza. Me `resample_freq`
parazgjedhje (`5min`) dhe dataset-in Kaggle, ka ~577 rreshta "live"/klient:
me `--rounds 80` kjo jep ~7 rreshta/raund (mbi `min_new_samples_to_train=6`,
OK); me `--rounds 200` jep vetëm ~2.9/raund (**NËN** 6) -- axis-i (1) i
kontrolluesit ("new\_samples < min -> skip train") e kapërcen trajnimin **NE
ÇDO RAUND, GJITHË XHIRIMIN**, pa asnjë gabim -- thjesht një CSV me
`rounds_trained=0` kudo, që duket si rezultat "normal" nëse s'e kontrollon
shprehimisht atë kolonë (RMSE ende llogaritet, nga peshat e warm-start-it, pa
asnjë trajnim/dërgim real -- numra që duken aq sa duhen po janë bosh).

`prepare_clients()` tani e zbulon vetë këtë (për politikën `adaptive`, pa
`--ablate-train-skip`) dhe hedh `ValueError` menjëherë, me numrin e saktë të
`--rounds` maksimale të sigurta për setup-in tënd -- pra nuk mund të të ndodhë
në heshtje më. Nëse të duhen vërtet shumë raunde (p.sh. 200, për ta ekspozuar
boshtin e çaktivizimit -- shih README\_OVERLEAF.md §3.3 te paketa e punimit),
zgjidhja është të
zvogëlosh `--resample-freq` (më shumë rreshta gjithsej), jo thjesht të rrisësh
`--rounds`:

```bash
# E VERIFIKUAR vetë -- funksionon, trajnon/dergon realisht, dhe battery i
# klientëve bateri-të-kufizuar bie deri ne 0% (boshti 5 EKSPOZOHET plotësisht,
# model_active=False per shumicen e raundeve):
python run_experiment.py --data data/iot_telemetry_data.csv --policy adaptive \
    --rounds 200 --resample-freq 2min --round-minutes 14.41 --run-name adaptive_200r
python run_experiment.py --data data/iot_telemetry_data.csv --policy non_adaptive \
    --rounds 200 --resample-freq 2min --round-minutes 14.41 --run-name nonadaptive_200r
```

(`--round-minutes 14.41` = sa minuta reale përfaqëson mesatarisht 1 raund me
këtë setup, `2min` x ~7.2 rreshta/raund -- përdoret vetëm nga modeli i
baterisë, mbaje konsistente me `--resample-freq`/`--rounds` e zgjedhura.)
3. Xhiro `analyze_results.py` për figura/tabela.
4. Opsionale, por e rekomanduar për punim: **ablacione** -- tani të integruara
   drejtpërdrejt si flamuj CLI (s'ka nevojë të editosh `controller.py`):
   `--ablate-train-skip`, `--ablate-epoch-adapt`, `--ablate-send-hold`,
   `--ablate-compression`, `--ablate-deactivation`, `--ablate-lite-switch`
   (secili çaktivizon VETËM atë dimension të vetëm, 5 të tjerët mbeten plotësisht
   adaptivë). Shembull:
   ```bash
   python run_experiment.py --data data/iot_telemetry_data.csv --policy adaptive \
       --rounds 80 --ablate-compression --run-name adaptive_80r_ablate_compression
   ```
   Krahaso `outputs/adaptive_80r.csv` (pa ablacion) me secilin `outputs/adaptive_80r_ablate_*.csv`
   për të izoluar kontributin e secilit dimension.
5. Më dërgo CSV-të e outputit (`outputs/*.csv`) dhe/ose figurat -- unë i
   analizoj, i kthej në tabela/figura gati-për-punim, dhe shkruaj draft-in.

## 10. Struktura e kodit

```
src/config.py        -- presets, parametra eksperimenti, profilet e pajisjeve, pragjet
src/data_loader.py    -- ngarkim/pastrim/ndarje-sipas-device CSV
src/features.py        -- lag/rolling/ciklike (full) dhe minimal (lite)
src/models.py           -- regresion linear FedAvg-compatible (gradient descent, clipping)
src/device_state.py      -- bateri/CPU/RAM/rrjet të simuluara, evolim real i baterisë
src/drift.py               -- injektim + zbulim (EWMA) i concept drift
src/compression.py          -- kuantizim 8-bit + delta per update të kompresuar
src/controller.py             -- KONTRIBUTI QENDROR: 9 sinjale -> 6 vendime
src/fl_simulation.py            -- orkestrimi FedAvg raund-pas-raundi
run_experiment.py                -- CLI
analyze_results.py                -- krahasim (mesatare mbi klientë) + figura + tabelë përmbledhëse
plot_drift_comparison.py             -- figura e krahasimit të drift-it (vetëm klienti i targetuar, shih §4)
tests/                                 -- teste automatike (fixture i vogël sintetik)
```

## 11. Citimi

Nëse e përdorni këtë kod për punë akademike, ju lutem citoni punimin:

```bibtex
@article{shkurti2026fedadaptive,
  title   = {Federated Adaptive Forecasting for Heterogeneous IoT Sensor Networks Under Concept Drift},
  author  = {Shkurti, Lamir and TODO: Co-Author},
  journal = {TODO: pas pranimit, shto journal-in/DOI-n e MDPI},
  year    = {2026}
}
```

## 12. Licenca

MIT -- shih skedarin `LICENSE`. Dataset-i (`data/iot_telemetry_data.csv`,
Kaggle, shkarkohet vetë sipas `data/README.md`) **NUK** është pjesë e kësaj
licence dhe mbetet nën kushtet e Kaggle-t/autorit origjinal (shih §3).
# adaptive-fl-concept-drift
