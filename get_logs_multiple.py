import simpler_env
from simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict

import sys
import time

import json

TASKS = ["widowx_carrot_on_plate", "widowx_put_eggplant_in_basket", "widowx_stack_cube"]

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

action_log = dict()

for task_name in TASKS:
    print(f"Task: {task_name}")
    env = simpler_env.make(task_name)

    seeds = generate_seeds(iterations)

    successes = []

    action_log[task_name] = action_log.get(task_name, dict())
    action_log[task_name]['iterations'] = iterations
    action_log[task_name]['total'] = action_log[task_name].get('total', 0)
    action_log[task_name]['sr'] = action_log[task_name].get('sr', 0)

    for i in range(iterations):
        print(f'Starting iteration: {i}')
        action_log[task_name][i] = action_log[task_name].get(i, dict())
        action_log[task_name][i]['seed'] = seeds[i]
        action_log[task_name][i]['actions'] = dict()
        obs, reset_info = env.reset(seed=seeds[i])
        instruction = env.get_language_instruction()
        frames = [get_image_from_maniskill2_obs_dict(env, obs), get_image_from_maniskill2_obs_dict(env, obs), get_image_from_maniskill2_obs_dict(env, obs)]
        done, truncated = False, False
        j = 0
        while (not done) and (j < 300):
            image = get_image_from_maniskill2_obs_dict(env, obs)
            frames.append(image)
            print(f"I: {i}; Action: {j}: ", end='')
            import requests
            import json_numpy
            json_numpy.patch()
            import numpy as np
            np.set_printoptions(6)
            action = requests.post(
                "http://0.0.0.0:8000/act",
                json={"image1": frames[-1], "image2": frames[-2], "image3": frames[-3], "image4": frames[-4], "instruction": instruction, "unnorm_key": "bridge_orig"}
            ).json()
            action = np.concatenate([action["world_vector"], action["rot_axangle"], action["gripper"]])
            grep = action[-1]
            action = np.array(action) * 1
            action[-1] = grep
            action_log[task_name][i]['actions'][j] = list(action)

            obs, reward, done, truncated, info = env.step(action)
            print(f"reward: {reward}, done: {done}, truncated: {truncated}; ", end='')
            print(f"{action}")

            j += 1

        successes.append(done)
        action_log[task_name][i]['done'] = done
        action_log[task_name][i]['truncated'] = truncated
        action_log[task_name][i]['total'] = j
        action_log[task_name]['total'] = action_log[task_name].get('total', 0) + j

        path_to_gif = f'./gifs_logs_multiple/{task_name + "_" + str(i)}_{time.time()}.{EXTENSION}'
        print(f"{path_to_gif=}")
        action_log[task_name]['path_to_gif'] = path_to_gif
        display_images(frames, path_to_gif)


    successes_count = 0
    for success in successes:
        successes_count += success
    success_rate = successes_count / iterations * 100
    print(f'{task_name} success rate: {success_rate}%')
    write_log(f'{task_name} {time.time()}.log', f'{task_name} success rate: {success_rate}%')
    action_log[task_name]['sr'] = success_rate


with open('actions_multiple.json', 'w') as f:
    json.dump(action_log, f, indent=4)
