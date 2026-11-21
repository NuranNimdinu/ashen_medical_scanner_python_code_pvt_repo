import time
import threading
import board
import busio
import neopixel_spi


class WS2812BThread(threading.Thread):
    def __init__(
        self,
        width=4,
        height=4,
        serpentine=True,
        brightness=0.2,
        spi_bus=1,
        spi=None,
        pixel_order=neopixel_spi.GRB,
        frequency=6_400_000,
        reset_time=300e-6,
        bit0=0b01100000,
        bit1=0b01111000,
    ):
        super().__init__(daemon=True)
        self.width = width
        self.height = height
        self.num_leds = width * height
        self.serpentine = serpentine

        # Create SPI bus if one was not passed in
        if spi is not None:
            self._spi = spi
            self._owns_spi = False
        elif spi_bus == 1:
            self._spi = busio.SPI(board.SCLK_1, MOSI=board.MOSI_1, MISO=board.MISO_1)
            self._owns_spi = True
        elif spi_bus == 0:
            self._spi = busio.SPI(board.SCLK, MOSI=board.MOSI, MISO=board.MISO)
            self._owns_spi = True
        else:
            raise ValueError("spi_bus must be 0 or 1 (or pass an explicit busio.SPI object).")

        # Initialize NeoPixel over SPI with SPI1-safe pulse timings
        self.pixels = neopixel_spi.NeoPixel_SPI(
            self._spi,
            self.num_leds,
            brightness=brightness,
            auto_write=False,
            pixel_order=pixel_order,
            frequency=frequency,
            reset_time=reset_time,
            bit0=bit0,
            bit1=bit1,
        )

        # Thread control flags
        self._stop_event = threading.Event()
        self._update_event = threading.Event()
        self._mode_lock = threading.Lock()

        # State variables (frame-based so both solid colors and 2D patterns work seamlessly)
        self._black_frame = [(0, 0, 0)] * self.num_leds
        self._mode = "CLEAR"
        self._current_frame = list(self._black_frame)
        self._previous_static_frame = None
        self._brightness = brightness
        self._interval = 0.5
        self._pulses_left = 0
        self._infinite_blink = False

    def xy_to_index(self, x, y):
        """Maps 2D (x, y) coordinates to the 1D LED strip index."""
        if self.serpentine and (y % 2 == 1):
            return y * self.width + (self.width - 1 - x)
        return y * self.width + x

    @staticmethod
    def _hex_to_rgb(hex_color):
        """Converts an integer hex code (e.g., 0xFF0000) to an RGB tuple (255, 0, 0)."""
        r = (hex_color >> 16) & 0xFF
        g = (hex_color >> 8) & 0xFF
        b = hex_color & 0xFF
        return (r, g, b)

    def _normalize_color(self, color):
        """Accepts either an integer hex color (0xFF0000) or an (r, g, b) tuple."""
        if isinstance(color, int):
            return self._hex_to_rgb(color)
        if isinstance(color, (tuple, list)) and len(color) == 3:
            return (int(color[0]), int(color[1]), int(color[2]))
        raise ValueError("Color must be an integer hex value (e.g. 0xFF0000) or an (R, G, B) tuple.")

    def _write_frame(self, frame, brightness):
        """Writes a full frame to the SPI NeoPixel buffer and pushes it out."""
        if self.pixels.brightness != brightness:
            self.pixels.brightness = brightness
        for i, rgb in enumerate(frame):
            self.pixels[i] = rgb
        self.pixels.show()

    def set_brightness(self, brightness):
        """Updates matrix brightness (0.0 to 1.0) thread-safely."""
        brightness = max(0.0, min(1.0, float(brightness)))
        with self._mode_lock:
            self._brightness = brightness
            self._update_event.set()

    def set_static_color(self, color):
        """Fills the entire matrix with a solid color and saves it as the baseline state."""
        rgb = self._normalize_color(color)
        frame = [rgb] * self.num_leds
        with self._mode_lock:
            self._mode = "STATIC"
            self._current_frame = frame
            self._previous_static_frame = list(frame)
            self._update_event.set()

    def set_pixel(self, x, y, color, show=True):
        """
        Updates a single (x, y) pixel on the 4x4 matrix.
        Set show=False to batch multiple pixel changes, then call show_frame().
        """
        if not (0 <= x < self.width and 0 <= y < self.height):
            return
        rgb = self._normalize_color(color)
        idx = self.xy_to_index(x, y)
        with self._mode_lock:
            base = (
                list(self._previous_static_frame)
                if self._previous_static_frame is not None
                else list(self._black_frame)
            )
            base[idx] = rgb
            self._previous_static_frame = base
            if show:
                self._mode = "STATIC"
                self._current_frame = list(base)
                self._update_event.set()

    def show_frame(self):
        """Displays any pixels previously staged with set_pixel(..., show=False)."""
        with self._mode_lock:
            frame = (
                list(self._previous_static_frame)
                if self._previous_static_frame is not None
                else list(self._black_frame)
            )
            self._mode = "STATIC"
            self._current_frame = frame
            self._update_event.set()

    def set_matrix(self, grid):
        """
        Atomically updates the full matrix from a 2D list [y][x] of colors
        (hex ints or (R, G, B) tuples) and saves it as the baseline static state.
        """
        if len(grid) != self.height or any(len(row) != self.width for row in grid):
            raise ValueError(f"Expected a {self.height}x{self.width} grid.")

        frame = list(self._black_frame)
        for y in range(self.height):
            for x in range(self.width):
                frame[self.xy_to_index(x, y)] = self._normalize_color(grid[y][x])

        with self._mode_lock:
            self._mode = "STATIC"
            self._current_frame = frame
            self._previous_static_frame = list(frame)
            self._update_event.set()

    def blink(self, color, interval, pulses=0):
        """
        Blinks the entire matrix in a solid color.
        interval: half-cycle duration in seconds (on = interval, off = interval).
        pulses: number of blinks; 0 or None means infinite.
        """
        rgb = self._normalize_color(color)
        frame = [rgb] * self.num_leds
        with self._mode_lock:
            self._mode = "BLINK"
            self._current_frame = frame
            self._interval = float(interval)
            self._infinite_blink = (pulses is None or pulses <= 0)
            self._pulses_left = 0 if self._infinite_blink else int(pulses)
            self._update_event.set()

    def stop_blink(self):
        """
        Stops blinking immediately and restores the previous static color/pattern,
        or clears the matrix if none was active.
        """
        with self._mode_lock:
            if self._previous_static_frame is not None:
                self._mode = "STATIC"
                self._current_frame = list(self._previous_static_frame)
            else:
                self._mode = "CLEAR"
                self._current_frame = list(self._black_frame)
            self._update_event.set()

    def clear(self):
        """Turns off all LEDs and clears any remembered static color/pattern."""
        with self._mode_lock:
            self._mode = "CLEAR"
            self._current_frame = list(self._black_frame)
            self._previous_static_frame = None
            self._update_event.set()

    def cleanup(self):
        """Safely stops the background thread, clears the matrix, and releases SPI."""
        self._stop_event.set()
        self._update_event.set()

        if self.is_alive():
            self.join()

        self.pixels.fill((0, 0, 0))
        self.pixels.show()
        if hasattr(self.pixels, "deinit"):
            self.pixels.deinit()
        if self._owns_spi:
            self._spi.deinit()

    def run(self):
        """Main background loop."""
        while not self._stop_event.is_set():
            with self._mode_lock:
                mode = self._mode
                frame = list(self._current_frame)
                brightness = self._brightness
                interval = self._interval
                infinite = self._infinite_blink
                self._update_event.clear()

            if mode == "CLEAR":
                self._write_frame(self._black_frame, brightness)
                self._update_event.wait()

            elif mode == "STATIC":
                self._write_frame(frame, brightness)
                self._update_event.wait()

            elif mode == "BLINK":
                # ON Phase
                self._write_frame(frame, brightness)
                if self._update_event.wait(timeout=interval):
                    continue

                # OFF Phase
                self._write_frame(self._black_frame, brightness)
                if self._update_event.wait(timeout=interval):
                    continue

                # Decrement pulses after a complete ON/OFF cycle
                if not infinite:
                    with self._mode_lock:
                        if self._mode == "BLINK":
                            self._pulses_left -= 1
                            if self._pulses_left <= 0:
                                if self._previous_static_frame is not None:
                                    self._mode = "STATIC"
                                    self._current_frame = list(self._previous_static_frame)
                                else:
                                    self._mode = "CLEAR"
                                    self._current_frame = list(self._black_frame)