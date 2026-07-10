# Flashing the ESP32-S3 boards

Two ESP32-S3-DevKitC-1-N16R8 boards: board A transmits (`csi_send`), board B extracts
CSI (`csi_recv`) and streams it over USB serial to the collector.

Firmware source: [espressif/esp-csi](https://github.com/espressif/esp-csi), cloned to
`~/esp/esp-csi`. ESP-IDF lives at `~/esp/esp-idf`.

## 0. One-time host setup (WSL2 Ubuntu)

```bash
sudo apt-get install -y git wget flex bison gperf cmake ninja-build ccache \
    libffi-dev libssl-dev dfu-util libusb-1.0-0 python3-venv python3-pip

cd ~/esp/esp-idf
./install.sh esp32s3
```

Every new shell that uses idf.py needs:

```bash
source ~/esp/esp-idf/export.sh
```

## 1. Build both images (no board required)

```bash
source ~/esp/esp-idf/export.sh

cd ~/esp/esp-csi/examples/get-started/csi_send
idf.py set-target esp32s3 build

cd ~/esp/esp-csi/examples/get-started/csi_recv
idf.py set-target esp32s3 build
```

Build artifacts land in each example's `build/` directory (`bootloader.bin`,
`partition-table.bin`, and the app `.bin`).

## 2. Attach USB serial to WSL2 (usbipd-win)

USB devices are owned by Windows. From an elevated PowerShell on the Windows side:

```powershell
winget install usbipd            # once
usbipd list                      # find the BUSID of "USB JTAG/serial debug unit"
usbipd bind --busid <BUSID>      # once per device
usbipd attach --wsl --busid <BUSID>   # every replug
```

Back in WSL:

```bash
ls /dev/ttyACM*                  # should show /dev/ttyACM0
sudo usermod -aG dialout $USER   # once, then re-login
```

Plug in ONE board at a time so you always know which port is which.

## 3. Flash

```bash
# board A = TX
cd ~/esp/esp-csi/examples/get-started/csi_send
idf.py -p /dev/ttyACM0 flash monitor

# board B = RX (plug in after unplugging A, or use the second ttyACM)
cd ~/esp/esp-csi/examples/get-started/csi_recv
idf.py -p /dev/ttyACM0 flash monitor
```

`monitor` on the RX board should print `CSI_DATA,...` lines once the TX board is
powered (any USB wall adapter works for the TX after flashing). Ctrl+] exits monitor.

Label the boards physically (TX / RX) after flashing.

## Fallback: flash from Windows if usbipd misbehaves

The pre-built .bin files can be flashed with esptool from the Windows side (COM port
from Device Manager). Flash offsets come from the build output; typical layout:

```powershell
pip install esptool
esptool --chip esp32s3 --port COM5 write_flash `
    0x0 bootloader.bin 0x8000 partition-table.bin 0x10000 <app>.bin
```

Run `idf.py build` once in WSL and copy `build/*.bin` to Windows, or check the exact
offsets in `build/flash_args`.

## Plan B: single-radio mode

If only one board survives shipping, `examples/get-started/csi_recv_router` extracts
CSI from traffic to the home router (ping flood): flash it with WiFi credentials in
menuconfig. Noisier than the dedicated TX/RX pair, but keeps development moving.

## 4. Hook up the collector

```bash
uv run python -m collector.collector --port /dev/ttyACM0 --room bedroom \
    --node-positions "TX:(x,y) RX:(x,y)" --occupants ramiro dog1 dog2
```

Then open the dashboard Live page. Placement rules for the bedroom deployment:
nodes on opposite sides of the bed, mattress height, 2-6 m apart, bed inside the
TX-RX path, away from the TV and large metal objects.
