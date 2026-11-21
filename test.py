# from rpi_libs.SPIStepperControllerLib import SPIStepperController
# import time
# with SPIStepperController(
#     spi_bus=0,
#     dir_pin=24,
#     limit_switch_pin=4,
#     rotations_per_mm=0.5,
#     steps_per_rev=800,         # Confirm A4988 is wired to 1/4: MS1=L, MS2=H, MS3=L
#     max_rate_mm_s=25.0,
#     max_travel_mm=210.0,
#     limit_pushed_state=0,
#     forward_dir_state=0,
#     accel_mm_s2=10.0,
#     start_rate_mm_s=0.5,
#     profile="trapezoidal",          # "trapezoidal" or "scurve"
#     max_spi_hz=250_000,
# ) as axis:
#     axis.reset_to_initial(rate_mm_s=8.0)   # Safe un-ramped homing speed

#     while 1:
#         axis.travel_to_position(position_mm=190.0, rate_mm_s=20.0)
#         time.sleep(0.5)
#         axis.travel_to_position(position_mm=20.0, rate_mm_s=20.0)
#         time.sleep(0.5)


import time
from rpi_libs.ws2812_led_indicator import WS2812BThread
 

matrix = WS2812BThread(
    width=4,
    height=4,
    serpentine=True,
    brightness=1,
    spi_bus=1,
)
matrix.start()

try:
    # 1. Original solid-color & blink workflow
    matrix.set_static_color(0xFFFFFF)  # White illumination
    time.sleep(2)

    matrix.blink(0x00FF00, interval=0.25, pulses=3)  # Blink green 3x, then auto-restore white
    time.sleep(2)

    # 2. Batch individual 2D pixel updates (only 1 SPI write on show_frame)
    matrix.clear()
    matrix.set_pixel(0, 0, (255, 0, 0), show=False)
    matrix.set_pixel(3, 0, (0, 255, 0), show=False)
    matrix.set_pixel(0, 3, (0, 0, 255), show=False)
    matrix.set_pixel(3, 3, (255, 255, 0), show=False)
    matrix.show_frame()
    time.sleep(2)

    # 3. Set a full 4x4 lighting pattern (e.g., outer ring on, inner 2x2 off)
    W = 0xFFFFFF
    O = 0xFF0000
    matrix.set_matrix([
        [W, W, W, W],
        [W, O, O, W],
        [W, O, O, W],
        [W, W, W, W],
    ])
    time.sleep(2)

finally:
    matrix.cleanup()