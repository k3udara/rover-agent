import os
from pathlib import Path

from controller import Robot

SAVE_EVERY_SECONDS = 0.064
JPEG_QUALITY = 90

image_file = Path(__file__).resolve().parents[3] / "runs" / "webots_overhead.jpg"
temporary_file = image_file.with_name("webots_overhead.part.jpg")
image_file.parent.mkdir(parents=True, exist_ok=True)

robot = Robot()
time_step = int(robot.getBasicTimeStep())
camera = robot.getDevice("overhead")
camera.enable(time_step)
steps_between_saves = max(1, round(SAVE_EVERY_SECONDS * 1000 / time_step))

step = 0
while robot.step(time_step) != -1:
    step += 1
    if step % steps_between_saves == 0 and camera.saveImage(str(temporary_file), JPEG_QUALITY) == 0:
        os.replace(temporary_file, image_file)
