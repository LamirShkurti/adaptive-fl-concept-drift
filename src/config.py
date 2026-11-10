
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# Dataset preset
# ---------------------------------------------------------------------------

@dataclass
class DatasetPreset:
    name: str
    time_col: str
    device_col: Optional[str]
    target_col: str
    feature_cols: List[str]
    time_is_unix_seconds: bool = False
    time_format: Optional[str] = None


PRESET_IOT_TELEMETRY = DatasetPreset(
    name="iot_telemetry",
    time_col="ts",
    device_col="device",
    target_col="temp",
    feature_cols=["temp", "humidity", "co", "lpg", "smoke"],
    time_is_unix_seconds=True,
)

PRESETS = {"iot_telemetry": PRESET_IOT_TELEMETRY}

# Emri i skedarit qe duhet vendosur te data/ (shih data/README.md).
EXPECTED_DATA_FILENAME = "iot_telemetry_data.csv"


# ---------------------------------------------------------------------------
# Parametrat e eksperimentit
# ---------------------------------------------------------------------------

@dataclass
class ExperimentConfig:
    preset: str = "iot_telemetry"

    # Resample-i per nje "raund" FL -- te dhenat origjinale jane ~1-7 sekonda,
    # shume te shpeshta per nje raund kuptimplote trajnimi; agregojme ne
    # dritare me te gjata (p.sh. 5min) qe çdo raund te pasqyroje nje sasi
    # realiste te dhenash te reja te mbledhura mes dy raundeve.
    resample_freq: str = "5min"

    horizon_steps: int = 12          # 12 x 5min = 1 ore parashikim, si ne punimin e meparshem
    n_lags: int = 6
    rolling_windows: List[int] = field(default_factory=lambda: [3, 6, 12])

    train_frac: float = 0.60
    val_frac: float = 0.15
    test_frac: float = 0.25          # pjesa e fundit kronologjike = "live" rounds te simulimit FL

    random_state: int = 42
    output_dir: str = "outputs"

    def get_preset(self) -> DatasetPreset:
        return PRESETS[self.preset]


# ---------------------------------------------------------------------------
# Profilet e pajisjeve (heterogjeniteti i klienteve FL)
# ---------------------------------------------------------------------------
#
# Secilit nga klientet reale (device-at e dataset-it) i caktojme nje "klase
# pajisjeje" simuluese per CPU/RAM/rrjet/bateri. Numrat jane kalibruar nga
# shperndarjet REALE te matura ne UseCase1_R30_6C (RPi3/4/5/6, MiniPC, Laptop)
# dhe nga profilimi Arduino MKR1000 (ARMv6-M, Cortex-M0+, 48MHz) qe perdoret
# per modelin "lite" ne punimin e parashikimit te temperatures. Qellimi: mos
# shpikim numra arbitrare -- çdo range poshte lidhet me nje burim konkret.

@dataclass
class DeviceClassProfile:
    label: str
    # CPU: % utilizim mesatar gjate nje raundi trajnimi lokal (mean, std) --
    # kalibruar nga UseCase1 'cpu' column (RPI range ~85-92%, MiniPC/Laptop ~23-32%
    # ne raunde te lehta, deri >90% ne raunde te renda).
    cpu_busy_mean: float
    cpu_busy_std: float
    # Temperatura CPU (°C) -- UseCase1 'cpu_temp' column.
    cpu_temp_mean: float
    cpu_temp_std: float
    # RAM e disponueshme (MB) -- UseCase1 'ram' column (RPi ~2.7-3.1 GB total
    # sistem; MiniPC/Laptop ~6.4 GB). Perfaqeson RAM TE LIRE, jo totale.
    ram_free_mb_mean: float
    ram_free_mb_std: float
    # Bandwidth gjate nje raundi (Mbps) -- UseCase1 'bandwidth_total' column.
    bandwidth_mbps_mean: float
    bandwidth_mbps_std: float
    # Ping/latence (ms) -- UseCase1 'ping' column (0 kur eshte lokal, deri ~43ms).
    latency_ms_mean: float
    latency_ms_std: float
    # Kapaciteti i baterise (mAh) dhe rrjedhja e rrymes (mA) gjate trajnimit
    # aktiv kundrejt idle -- te derivuara nga profilimi Arduino MKR1000 (48MHz,
    # ~4.3-5.4 us/parashikim, ARMv6-M software float) shkallezuar per klasen e
    # pajisjes. Pajisjet jo-bateri (MiniPC/Laptop) marrin kapacitet shume te
    # madh = praktikisht pa kufizim energjie, qe pasqyron realitetin.
    battery_capacity_mah: float
    current_draw_idle_ma: float
    current_draw_training_ma: float
    # Nese pajisja eshte "bateri-e-kufizuar" fare (per raportim/analize).
    battery_constrained: bool = True


# cpu_busy/cpu_temp/ram/bandwidth/latency jane llogaritur direkt (mean, stdev)
# nga te 4 raundet UseCase1_R30_6C (2 Adaptive + 2 NOAdaptive, gjithsej 123
# rreshta/pajisje) ne server/*.xlsx te projektit FL. Keto NUK jane shpikur --
# jane statistika reale te 6 pajisjeve (Laptop, MiniPC, RPI4, RPI5, RPI7, RPI8).
# battery_capacity_mah / current_draw_* MBETEN supozime inxhinierike (UseCase1
# nuk mati bateri -- ato pajisje ishin te lidhura ne prize/power-bank), bazuar
# ne fletore te dhenash tipike te publikuara (Raspberry Pi 4/5 power, power-bank
# 10-20k mAh) dhe shkallezuar kunder profilit real Arduino MKR1000 (Cortex-M0+,
# 48MHz, ~4.3-5.4 us/parashikim, ~204-259 cikle -- punimi i parashikimit te
# temperatures, Seksioni 4.5). Deklarohen shprehimisht si supozim n'punim.
DEVICE_CLASSES: Dict[str, DeviceClassProfile] = {
    # RPI4/5/7/8 mesatarizuar (jane shume afer njeri-tjetrit ne te 4 raundet
    # reale: cpu ~68-69%, ram-free ~2.5GB, bandwidth ~0.5-0.65 Mbps).
    "rpi_class": DeviceClassProfile(
        label="Raspberry Pi 4/5-class node (battery-constrained)",
        cpu_busy_mean=68.8, cpu_busy_std=20.9,
        cpu_temp_mean=79.7, cpu_temp_std=4.7,
        ram_free_mb_mean=2511.9, ram_free_mb_std=265.6,
        bandwidth_mbps_mean=0.56, bandwidth_mbps_std=0.54,
        latency_ms_mean=16.0, latency_ms_std=20.0,
        battery_capacity_mah=10000.0,          # power-bank tipik
        current_draw_idle_ma=540.0,            # RPi4 idle ~2.7W @5V
        current_draw_training_ma=1200.0,       # RPi4 ngarkese e larte ~6W @5V
        battery_constrained=True,
    ),
    # MiniPC -- ne te dhenat reale eshte NE FAKT pajisja me e ngarkuar/me e
    # nxehte (cpu 77.9% mesatare, deri 102°C cpu_temp); e trajtojme si nyje
    # me bateri te madhe por me konsum te larte (p.sh. mini-PC SBC portativ).
    "minipc_class": DeviceClassProfile(
        label="Mini-PC-class node (battery-constrained, high load)",
        cpu_busy_mean=77.9, cpu_busy_std=15.7,
        cpu_temp_mean=90.2, cpu_temp_std=8.7,
        ram_free_mb_mean=3022.6, ram_free_mb_std=113.9,
        bandwidth_mbps_mean=0.92, bandwidth_mbps_std=0.66,
        latency_ms_mean=27.5, latency_ms_std=25.5,
        battery_capacity_mah=20000.0,
        current_draw_idle_ma=900.0,
        current_draw_training_ma=2200.0,
        battery_constrained=True,
    ),
    # Laptop -- mains-powered, ngarkesa me e ulet ne te 4 raundet reale
    # (cpu ~26%, cpu_temp ~63°C) -- pa kufizim real energjie.
    "laptop_class": DeviceClassProfile(
        label="Laptop-class node (mains-powered)",
        cpu_busy_mean=26.25, cpu_busy_std=3.76,
        cpu_temp_mean=63.3, cpu_temp_std=3.1,
        ram_free_mb_mean=6224.3, ram_free_mb_std=242.0,
        bandwidth_mbps_mean=0.26, bandwidth_mbps_std=0.34,
        latency_ms_mean=0.0, latency_ms_std=0.5,
        battery_capacity_mah=1e9,
        current_draw_idle_ma=0.0,
        current_draw_training_ma=0.0,
        battery_constrained=False,
    ),
}

# Cakto nje klase pajisjeje secilit nga 3 device-at e dataset-it Kaggle (qe
# jane identifikuar vetem me MAC-adresa, pa metadhena vendndodhjeje/klase).
# Caktimi eshte NJE SKENAR SIMULIMI I DEKLARUAR SHPREHIMISHT (jo matje fizike
# e ketyre 3 pajisjeve konkrete Kaggle), por identik/riprodhueshem ne te
# gjitha eksperimentet (e njejta "farë"/seed -- shih fl_simulation.py).
DEVICE_CLASS_ASSIGNMENT_ORDER = ["rpi_class", "minipc_class", "laptop_class"]


# ---------------------------------------------------------------------------
# Pragjet per kontrolluesin adaptiv (6 vendimet)
# ---------------------------------------------------------------------------

@dataclass
class ControllerThresholds:
    # (1) Trajno apo kapërce raundin:
    min_battery_pct_to_train: float = 15.0
    min_new_samples_to_train: int = 6

    # (2) Epochs lokale (min/max), shkallezohen sipas buxhetit te CPU/baterise.
    epochs_min: int = 1
    epochs_max: int = 6

    # (3) Frekuenca e dergimit te update-it: nese battery/bandwidth jane te
    # dobeta, lejo te mbahet update lokalisht deri X raunde pa e dërguar.
    max_rounds_without_send: int = 3

    # (4) Prag per kompresim: nese bandwidth < kete (Mbps) ose battery < kete (%),
    # dergo update te kompresuar (kuantizim 8-bit + delta ndaj raundit te fundit).
    compress_if_bandwidth_below_mbps: float = 0.5
    compress_if_battery_below_pct: float = 30.0

    # (5) Çaktivizo modelin lokal (kthehu te parashikimi i fundit i njohur mire)
    # nese battery eshte kritike.
    deactivate_if_battery_below_pct: float = 8.0

    # (6) Kalo ne modelin "lite" (2-feature, si ne MKR1000) nese battery/CPU
    # jane vazhdimisht nen stres per disa raunde rresht.
    switch_to_lite_after_consecutive_stress_rounds: int = 3
    stress_cpu_pct: float = 85.0
    stress_battery_pct: float = 25.0

    # Zbulim concept-drift: rritje e papritur e gabimit lokal (EWMA ratio)
    # krahasuar me mesataren e dritares se fundit.
    drift_error_ratio_threshold: float = 1.6
    drift_ewma_alpha: float = 0.3


# ---------------------------------------------------------------------------
# Ablacione (per studimin "nje-bosht-ne-kohe" te kerkuar per punim)
# ---------------------------------------------------------------------------
#
# Secili flag, kur eshte True, e "fikset" ate bosht te vetem te kontrolluesit
# adaptiv ne sjelljen e tij baseline/jo-adaptive, DUKE LENE 5 boshtet e tjera
# plotesisht adaptive. Kjo lejon nje krahasim te paster "full-adaptive minus
# nje bosht" kunder "full-adaptive", per te izoluar kontributin e secilit
# bosht veç e veç (shih AdaptiveController ne controller.py per implementimin
# e sakte te secilit flag).
@dataclass
class AblationFlags:
    disable_train_skip: bool = False      # bosht (1): gjithmone trajno (nese ka te dhena)
    disable_epoch_adapt: bool = False     # bosht (2): epochs fikse (fixed_epochs_for_ablation)
    disable_send_hold: bool = False       # bosht (3): dergo gjithmone kur trajnon
    disable_compression: bool = False     # bosht (4): gjithmone "full" (asnjehere i kompresuar)
    disable_deactivation: bool = False    # bosht (5): modeli gjithmone aktiv
    disable_lite_switch: bool = False     # bosht (6): gjithmone varianti "full"
    fixed_epochs_for_ablation: int = 3    # perdoret vetem kur disable_epoch_adapt=True
