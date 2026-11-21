from __future__ import annotations

import gc
import math
import time
import warnings
import spidev
import RPi.GPIO as GPIO


class SPIStepperController:
    """
    Stall-hardened open-loop linear stepper controller for Raspberry Pi 3.
    Pre-synthesizes the full accel + cruise + decel SPI MOSI bitstream in RAM
    and dynamically scales oversampling so moves fit inside a single hardware
    SPI DMA transfer (<= 65,535 bytes on BCM2837) with zero mid-move pauses.
    """

    # BCM2837 SPI DMA hardware DLEN register max is 65,535 bytes; keep a small margin.
    _HW_DMA_MAX_BYTES = 65_000

    # Preferred oversampling when move easily fits in one DMA transfer
    _TARGET_OVERSAMPLE = 12
    _MIN_RAMP_OVERSAMPLE = 4

    # Safe minimum hardware SPI clock floors on RPi 3 (at 400 MHz core_freq)
    _MIN_SPI_HZ_BY_BUS = {
        0: 10_000,   # SPI0 hardware floor ~6.1 kHz
        1: 50_000,   # SPI1 hardware floor ~48.8 kHz
    }

    _DIR_SETUP_DELAY_S = 20e-6

    def __init__(
        self,
        spi_bus: int,
        dir_pin: int,
        limit_switch_pin: int,
        rotations_per_mm: float,
        max_rate_mm_s: float,
        max_travel_mm: float,
        limit_pushed_state: int,
        forward_dir_state: int,
        steps_per_rev: int = 200,
        accel_mm_s2: float = 100.0,
        start_rate_mm_s: float = 0.5,
        profile: str = "scurve",
        min_pulse_us: float = 5.0,
        reverse_dwell_s: float = 0.025,
        max_spi_hz: int = 250_000,
        spi_device: int = 0,
    ):
        if spi_bus not in self._MIN_SPI_HZ_BY_BUS:
            raise ValueError("Raspberry Pi 3 only supports spi_bus 0 or 1.")
        if limit_pushed_state not in (0, 1) or forward_dir_state not in (0, 1):
            raise ValueError("limit_pushed_state and forward_dir_state must be 0 or 1.")
        if (
            rotations_per_mm <= 0
            or max_rate_mm_s <= 0
            or max_travel_mm <= 0
            or steps_per_rev <= 0
            or accel_mm_s2 <= 0
            or start_rate_mm_s <= 0
            or min_pulse_us <= 0
            or reverse_dwell_s < 0
            or max_spi_hz <= 0
        ):
            raise ValueError("Mechanical, rate, acceleration, timing, and SPI parameters must be positive.")
        if profile not in ("scurve", "trapezoidal"):
            raise ValueError("profile must be 'scurve' or 'trapezoidal'.")

        self.spi_bus = spi_bus
        self.spi_device = spi_device
        self.dir_pin = dir_pin
        self.limit_switch_pin = limit_switch_pin
        self.rotations_per_mm = float(rotations_per_mm)
        self.steps_per_rev = int(steps_per_rev)
        self.steps_per_mm = self.rotations_per_mm * self.steps_per_rev
        self.max_rate_mm_s = float(max_rate_mm_s)
        self.max_travel_mm = float(max_travel_mm)
        self.limit_pushed_state = int(limit_pushed_state)
        self.forward_dir_state = int(forward_dir_state)
        self._backward_dir_state = 1 - self.forward_dir_state

        self.accel_mm_s2 = float(accel_mm_s2)
        self.start_rate_mm_s = float(start_rate_mm_s)
        self.profile = profile
        self.min_pulse_us = float(min_pulse_us)
        self.reverse_dwell_s = float(reverse_dwell_s)
        self.max_spi_hz = int(max_spi_hz)

        self._min_spi_hz = self._MIN_SPI_HZ_BY_BUS[spi_bus]
        self.kernel_bufsiz = self._read_kernel_bufsiz()
        self._single_dma_target_bytes = min(self._HW_DMA_MAX_BYTES, self.kernel_bufsiz - 256)
        if self.kernel_bufsiz < 65536:
            warnings.warn(
                f"Linux spidev.bufsiz is {self.kernel_bufsiz} bytes (< 65536). "
                "Add 'spidev.bufsiz=65536' to /boot/firmware/cmdline.txt (or /boot/cmdline.txt) "
                "and reboot to avoid mid-move SPI chunk pauses.",
                RuntimeWarning,
            )

        self._current_steps = 0
        self._current_dir: int | None = None
        self._current_spi_hz = 0
        self._is_homed = False

        # Initialize GPIO (BCM numbering)
        GPIO.setmode(GPIO.BCM)
        GPIO.setwarnings(False)
        GPIO.setup(self.dir_pin, GPIO.OUT, initial=self.forward_dir_state)
        self._current_dir = self.forward_dir_state

        pull_mode = GPIO.PUD_UP if self.limit_pushed_state == 0 else GPIO.PUD_DOWN
        GPIO.setup(self.limit_switch_pin, GPIO.IN, pull_up_down=pull_mode)

        # Initialize SPI
        self._spi = spidev.SpiDev()
        self._spi.open(self.spi_bus, self.spi_device)
        self._spi.mode = 0
        self._spi.bits_per_word = 8
        self._spi.lsbfirst = False

    @staticmethod
    def _read_kernel_bufsiz() -> int:
        try:
            with open("/sys/module/spidev/parameters/bufsiz", "r", encoding="ascii") as f:
                return int(f.read().strip())
        except OSError:
            return 4096

    @property
    def current_position_mm(self) -> float:
        return self._current_steps / self.steps_per_mm

    @property
    def is_homed(self) -> bool:
        return self._is_homed

    def is_limit_triggered(self) -> bool:
        return GPIO.input(self.limit_switch_pin) == self.limit_pushed_state

    def _set_direction(self, dir_state: int) -> None:
        if self._current_dir != dir_state:
            if self.reverse_dwell_s > 0.0:
                time.sleep(self.reverse_dwell_s)
            GPIO.output(self.dir_pin, dir_state)
            self._current_dir = dir_state
            time.sleep(self._DIR_SETUP_DELAY_S)

    def _set_spi_hz(self, spi_hz: int) -> None:
        if self._current_spi_hz != spi_hz:
            self._spi.max_speed_hz = spi_hz
            self._current_spi_hz = spi_hz

    def _choose_spi_params(
        self,
        peak_hz: float,
        move_duration_s: float,
        has_ramp: bool,
    ) -> tuple[int, int, int]:
        """
        Picks (bits_per_step, spi_hz, pulse_bits) to keep the entire move inside
        one uninterrupted hardware DMA transfer whenever possible.
        """
        n_min = max(2, math.ceil(self._min_spi_hz / peak_hz))
        n_max = max(2, int(self.max_spi_hz // peak_hz))

        if n_min > n_max:
            bits_per_step = n_min if (peak_hz * 2.0 < self._min_spi_hz) else 2
        else:
            # Largest oversampling that still fits total move in 1 DMA transfer
            est_peak_cycles = max(1.0, move_duration_s * peak_hz)
            n_dma = int((self._single_dma_target_bytes * 8) // est_peak_cycles)
            min_desired = self._MIN_RAMP_OVERSAMPLE if has_ramp else 2
            desired = min(self._TARGET_OVERSAMPLE, max(min_desired, n_dma))
            bits_per_step = min(n_max, max(n_min, desired))

        spi_hz = int(round(peak_hz * bits_per_step))
        desired_pulse_bits = max(1, math.ceil(self.min_pulse_us * 1e-6 * spi_hz))
        pulse_bits = min(bits_per_step // 2, desired_pulse_bits)
        return bits_per_step, spi_hz, pulse_bits

    @staticmethod
    def _stamp_pulse(buf: bytearray, bit_idx: int, pulse_bits: int) -> None:
        if pulse_bits == 1:
            buf[bit_idx >> 3] |= 0x80 >> (bit_idx & 7)
        else:
            for b in range(bit_idx, bit_idx + pulse_bits):
                buf[b >> 3] |= 0x80 >> (b & 7)

    @staticmethod
    def _compute_accel_bit_positions(
        accel_steps: int,
        f0: float,
        f_peak: float,
        accel_hz_s: float,
        spi_hz: int,
        min_spacing_bits: int,
        profile: str,
    ) -> tuple[list[int], int]:
        positions = [0] * accel_steps
        prev = 0

        if profile == "trapezoidal":
            f0_sq = f0 * f0
            two_a = 2.0 * accel_hz_s
            scale = spi_hz / accel_hz_s
            sqrt = math.sqrt

            for k in range(1, accel_steps):
                idx = int((sqrt(f0_sq + two_a * k) - f0) * scale + 0.5)
                if idx < prev + min_spacing_bits:
                    idx = prev + min_spacing_bits
                positions[k] = idx
                prev = idx

            total_bits = int((sqrt(f0_sq + two_a * accel_steps) - f0) * scale + 0.5)
        else:
            df = f_peak - f0
            f_avg = 0.5 * (f0 + f_peak)
            t_ramp = accel_steps / f_avg
            t_spi = t_ramp * spi_hz

            s_total = float(accel_steps)
            pi = math.pi
            sin = math.sin
            cos = math.cos
            c_sin = (t_ramp * df) / (2.0 * pi)
            t_f_avg = t_ramp * f_avg
            t_half_df = t_ramp * (0.5 * df)

            u = 0.0
            for k in range(1, accel_steps):
                ds_du = t_f_avg - t_half_df * cos(pi * u)
                u += 1.0 / ds_du
                if u > 1.0:
                    u = 1.0

                for _ in range(6):
                    pu = pi * u
                    err = s_total * u - c_sin * sin(pu) - k
                    if -1e-5 < err < 1e-5:
                        break
                    ds_du = t_f_avg - t_half_df * cos(pu)
                    u -= err / ds_du
                    if u < 0.0:
                        u = 0.0
                    elif u > 1.0:
                        u = 1.0

                idx = int(u * t_spi + 0.5)
                if idx < prev + min_spacing_bits:
                    idx = prev + min_spacing_bits
                positions[k] = idx
                prev = idx

            total_bits = int(t_spi + 0.5)

        if total_bits < prev + min_spacing_bits:
            total_bits = prev + min_spacing_bits
        return positions, total_bits

    def _build_ramped_bitstream(
        self,
        total_steps: int,
        cruise_hz: float,
        accel_hz_s: float,
        start_hz: float,
        profile: str,
    ) -> tuple[bytearray, int]:
        """Pre-builds the entire accel -> cruise -> decel bitstream before sending."""
        f0 = min(start_hz, cruise_hz)

        if total_steps == 1 or cruise_hz <= f0:
            accel_steps = 0
            decel_steps = 0
            cruise_steps = total_steps
            peak_hz = cruise_hz
        else:
            if profile == "scurve":
                full_ramp_steps = (math.pi / 4.0) * (cruise_hz * cruise_hz - f0 * f0) / accel_hz_s
            else:
                full_ramp_steps = (cruise_hz * cruise_hz - f0 * f0) / (2.0 * accel_hz_s)

            if 2.0 * full_ramp_steps <= total_steps:
                accel_steps = int(round(full_ramp_steps))
                if accel_steps * 2 > total_steps:
                    accel_steps = total_steps // 2
                decel_steps = accel_steps
                cruise_steps = total_steps - accel_steps - decel_steps
                peak_hz = cruise_hz
            else:
                accel_steps = total_steps // 2
                decel_steps = accel_steps
                cruise_steps = total_steps - (accel_steps * 2)
                if profile == "scurve":
                    peak_hz = math.sqrt(f0 * f0 + (4.0 * accel_hz_s * accel_steps) / math.pi)
                else:
                    peak_hz = math.sqrt(f0 * f0 + 2.0 * accel_hz_s * accel_steps)

        ramp_dur = (2.0 * accel_steps) / (0.5 * (f0 + peak_hz)) if accel_steps > 0 else 0.0
        move_dur = ramp_dur + (cruise_steps / peak_hz)

        bits_per_step, spi_hz, pulse_bits = self._choose_spi_params(
            peak_hz=peak_hz,
            move_duration_s=move_dur,
            has_ramp=(accel_steps > 0),
        )
        stamp = self._stamp_pulse

        out = bytearray()
        carry_byte = 0
        bit_phase = 0
        accel_positions: list[int] = []

        # --- Phase 1: Acceleration ---
        if accel_steps > 0:
            accel_positions, accel_total_bits = self._compute_accel_bit_positions(
                accel_steps, f0, peak_hz, accel_hz_s, spi_hz, bits_per_step, profile
            )
            full_bytes, bit_phase = divmod(accel_total_bits, 8)
            accel_buf = bytearray(full_bytes + (1 if bit_phase else 0))
            for pos in accel_positions:
                stamp(accel_buf, pos, pulse_bits)

            if full_bytes:
                out.extend(accel_buf[:full_bytes])
            carry_byte = accel_buf[full_bytes] if bit_phase else 0

        # --- Phase 2: Constant-Speed Cruise ---
        if cruise_steps > 0:
            rem_cruise = cruise_steps

            if bit_phase >= bits_per_step:
                lead_steps = min(
                    rem_cruise,
                    (8 - bit_phase + bits_per_step - 1) // bits_per_step,
                )
                lead_end = bit_phase + lead_steps * bits_per_step
                lead_full, new_phase = divmod(lead_end, 8)
                lead_buf = bytearray(lead_full + (1 if new_phase else 0))
                lead_buf[0] = carry_byte
                pos = bit_phase
                for _ in range(lead_steps):
                    stamp(lead_buf, pos, pulse_bits)
                    pos += bits_per_step

                if lead_full:
                    out.extend(lead_buf[:lead_full])
                carry_byte = lead_buf[lead_full] if new_phase else 0
                bit_phase = new_phase
                rem_cruise -= lead_steps

            if rem_cruise > 0:
                period_bytes = bits_per_step // math.gcd(bits_per_step, 8)
                period_bits = period_bytes * 8
                steps_per_period = period_bits // bits_per_step
                full_periods, tail_steps = divmod(rem_cruise, steps_per_period)

                if full_periods > 0:
                    tmp = bytearray(period_bytes + 1)
                    pos = bit_phase
                    for _ in range(steps_per_period):
                        stamp(tmp, pos, pulse_bits)
                        pos += bits_per_step

                    spill = tmp[period_bytes]
                    first_period = bytearray(tmp[:period_bytes])
                    first_period[0] |= carry_byte
                    out.extend(first_period)

                    if full_periods > 1:
                        repeat_period = bytearray(tmp[:period_bytes])
                        repeat_period[0] |= spill
                        out.extend(bytes(repeat_period) * (full_periods - 1))

                    carry_byte = spill

                if tail_steps > 0:
                    tail_end = bit_phase + tail_steps * bits_per_step
                    tail_full, new_phase = divmod(tail_end, 8)
                    tail_buf = bytearray(tail_full + (1 if new_phase else 0))
                    tail_buf[0] = carry_byte
                    pos = bit_phase
                    for _ in range(tail_steps):
                        stamp(tail_buf, pos, pulse_bits)
                        pos += bits_per_step

                    if tail_full:
                        out.extend(tail_buf[:tail_full])
                    carry_byte = tail_buf[tail_full] if new_phase else 0
                    bit_phase = new_phase

        # --- Phase 3: Mirrored Deceleration ---
        if decel_steps > 0:
            last_b = accel_positions[-1]
            decel_total_bits = bit_phase + last_b + pulse_bits + 1
            decel_buf = bytearray((decel_total_bits + 7) // 8)
            decel_buf[0] = carry_byte

            base = bit_phase + last_b
            for pos in reversed(accel_positions):
                stamp(decel_buf, base - pos, pulse_bits)

            out.extend(decel_buf)
        elif bit_phase > 0 or carry_byte:
            out.append(carry_byte)

        return out, spi_hz

    def _transmit_buffer(self, buf: bytearray | bytes, spi_hz: int) -> None:
        if not buf:
            return
        self._set_spi_hz(spi_hz)
        gc_enabled = gc.isenabled()
        if gc_enabled:
            gc.disable()
        try:
            self._spi.writebytes2(buf)
        finally:
            if gc_enabled:
                gc.enable()

    def reset_to_initial(
        self,
        rate_mm_s: float | None = None,
        check_interval_s: float = 0.002,
    ) -> None:
        """Blocking homing routine with a pre-built burst to avoid per-loop stutter."""
        if rate_mm_s is None:
            rate_mm_s = min(self.max_rate_mm_s * 0.25, max(self.start_rate_mm_s * 4.0, 2.0))
        if rate_mm_s <= 0 or rate_mm_s > self.max_rate_mm_s:
            raise ValueError(f"Homing rate must be between 0 and {self.max_rate_mm_s} mm/s.")

        if self.is_limit_triggered():
            self._current_steps = 0
            self._is_homed = True
            return

        self._set_direction(self._backward_dir_state)

        step_hz = rate_mm_s * self.steps_per_mm
        steps_by_time = int(round(step_hz * check_interval_s))
        steps_by_dist = int(round(0.05 * self.steps_per_mm))
        chunk_steps = max(1, min(steps_by_time, steps_by_dist))

        burst_buf, spi_hz = self._build_ramped_bitstream(
            total_steps=chunk_steps,
            cruise_hz=step_hz,
            accel_hz_s=1.0,
            start_hz=step_hz,
            profile="trapezoidal",
        )
        burst_bytes = bytes(burst_buf)
        self._set_spi_hz(spi_hz)

        max_homing_steps = int(math.ceil(self.max_travel_mm * 1.1 * self.steps_per_mm))
        steps_moved = 0

        read_pin = GPIO.input
        sw_pin = self.limit_switch_pin
        pushed = self.limit_pushed_state
        write = self._spi.writebytes2

        gc_enabled = gc.isenabled()
        if gc_enabled:
            gc.disable()
        try:
            while read_pin(sw_pin) != pushed:
                if steps_moved >= max_homing_steps:
                    raise RuntimeError("Homing failed: limit switch not triggered within max travel length.")
                write(burst_bytes)
                steps_moved += chunk_steps
        finally:
            if gc_enabled:
                gc.enable()

        self._current_steps = 0
        self._is_homed = True

    def travel_to_position(
        self,
        position_mm: float,
        rate_mm_s: float,
        accel_mm_s2: float | None = None,
        profile: str | None = None,
    ) -> None:
        """Blocking absolute move in mm with single-DMA smooth ramping."""
        if position_mm < 0.0 or position_mm > self.max_travel_mm:
            raise ValueError(f"Target position {position_mm} mm is outside [0, {self.max_travel_mm}] mm.")
        if rate_mm_s <= 0.0 or rate_mm_s > self.max_rate_mm_s:
            raise ValueError(f"Rate {rate_mm_s} mm/s is outside (0, {self.max_rate_mm_s}] mm/s.")

        use_accel = self.accel_mm_s2 if accel_mm_s2 is None else float(accel_mm_s2)
        if use_accel <= 0.0:
            raise ValueError("accel_mm_s2 must be positive.")

        use_profile = self.profile if profile is None else profile
        if use_profile not in ("scurve", "trapezoidal"):
            raise ValueError("profile must be 'scurve' or 'trapezoidal'.")

        target_steps = round(position_mm * self.steps_per_mm)
        delta_steps = target_steps - self._current_steps
        if delta_steps == 0:
            return

        direction = self.forward_dir_state if delta_steps > 0 else self._backward_dir_state
        self._set_direction(direction)

        move_buf, spi_hz = self._build_ramped_bitstream(
            total_steps=abs(delta_steps),
            cruise_hz=rate_mm_s * self.steps_per_mm,
            accel_hz_s=use_accel * self.steps_per_mm,
            start_hz=self.start_rate_mm_s * self.steps_per_mm,
            profile=use_profile,
        )
        self._transmit_buffer(move_buf, spi_hz)

        self._current_steps = target_steps

    def close(self) -> None:
        if self._spi is not None:
            self._spi.close()
            self._spi = None
        GPIO.cleanup([self.dir_pin, self.limit_switch_pin])

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()