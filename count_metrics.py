import simpler_env
from simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict

import sys
import time

TASKS = ["google_robot_pick_coke_can", "google_robot_move_near", "google_robot_open_drawer", "google_robot_close_drawer",
         "widowx_spoon_on_towel", "widowx_carrot_on_plate", "widowx_stack_cube", "widowx_put_eggplant_in_basket"]

EXTENSION = 'gif'


def read_seeds(seeds_filename):
    with open(seeds_filename, "r") as seeds_file:
        seeds = [int(i) for i in seeds_file.readlines()]
        return seeds


def write_log(log_filename, log_message):
    with open(log_filename, "w") as log_file:
        log_file.write(log_message)


def generate_seeds(seeds_count):
    seeds = [i for i in range(seeds_count)]
    return seeds


def display_images(images, save_path=None):
    """Create a GIF from a list of RGB images. Save to the given path if provided, else return HTML to view."""
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    from IPython.display import HTML

    fig = plt.figure()
    ims = []

    for image in images:
        im = plt.imshow(image, animated=True)
        ims.append([im])

    ani = animation.ArtistAnimation(
        fig, ims, interval=200, blit=True, repeat_delay=1000)

    if save_path:
        ani.save(save_path, writer='imagemagick')
        # Prevents the final frame from being displayed as a static image
        plt.close(fig)


if len(sys.argv) >= 2:
    iterations = int(sys.argv[1])
else:
    sys.stderr.write('No iterations count\n')
    sys.exit()

for task_name in TASKS:
    env = simpler_env.make(task_name)

    seeds = generate_seeds(iterations)

    successes = []

    for i in range(iterations):
        print(f'Starting iteration: {i}')
        obs, reset_info = env.reset(seed=seeds[i])
        instruction = env.get_language_instruction()
        frames = []
        done, truncated = False, False
        while not (done or truncated):
            image = get_image_from_maniskill2_obs_dict(env, obs)
            
            import requests
            import json_numpy
            json_numpy.patch()
            import numpy as np
            
            action = requests.post(
                "http://0.0.0.0:8000/act",
                json={"image": image, "instruction": instruction, "unnorm_key": "bridge_orig"}
            ).json()
            action = np.array(action)

            obs, reward, done, truncated, info = env.step(action)
            frames.append(image)
        successes.append(done)
        #display_images(frames, f'./{task_name} {time.time()}.{EXTENSION}')

    successes_count = 0
    for success in successes:
        successes_count += success
    success_rate = successes_count / iterations * 100
    print(f'{task_name} success rate: {success_rate}%')
    write_log(f'{task_name} {time.time()}.log', f'{task_name} success rate: {success_rate}%')
