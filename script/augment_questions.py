import os
import json

import openai
import habitat_sim
import numpy as np
import matplotlib.pyplot as plt


def make_cfg(settings):
    # simulator backend
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_id = settings["scene"]
    sim_cfg.scene_dataset_config_file = settings["scene_dataset_config_file"]

    # agent
    agent_cfg = habitat_sim.agent.AgentConfiguration()

    # In the 1st example, we attach only one sensor, a RGB visual sensor, to the agent
    rgb_sensor_spec = habitat_sim.CameraSensorSpec()
    rgb_sensor_spec.uuid = "color_sensor"
    rgb_sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
    rgb_sensor_spec.resolution = [settings["height"], settings["width"]]
    rgb_sensor_spec.position = [0.0, settings["sensor_height"], 0.0]
    rgb_sensor_spec.hfov = settings["hfov"]

    agent_cfg.sensor_specifications = [rgb_sensor_spec]

    return habitat_sim.Configuration(sim_cfg, [agent_cfg])


if __name__ == "__main__":


    question_file_path = 'generated_questions/generated_questions.json'
    augmented_question_file_path = 'generated_questions/augmented_generated_questions.json'
    tempt_save_dir = 'augment_question_tempt_save_dir'

    scene_dataset_config_file = '/home/hanyang/code/3d_project/explore-eqa/data/versioned_data/hm3d-0.2/hm3d/hm3d_annotated_basis.scene_dataset_config.json'
    dataset_path = '/home/hanyang/code/3d_project/explore-eqa/data/versioned_data/hm3d-0.2/hm3d'

    camera_height = 1.2
    img_width = 1280
    img_height = 1280
    hfov = 100


    os.makedirs(tempt_save_dir, exist_ok=True)

    question_file = json.load(open(question_file_path, 'r'))
    if os.path.exists(augmented_question_file_path):
        augmented_question_file = json.load(open(augmented_question_file_path, 'r'))
    else:
        augmented_question_file = []

    # filter out questions that have already been augmented
    question_file = [q for q in question_file if q['question_id'] not in [aq['question_id'] for aq in augmented_question_file]]

    questions_per_scene = {}
    for q in question_file:
        if q['episode_history'] not in questions_per_scene:
            questions_per_scene[q['episode_history']] = []
        questions_per_scene[q['episode_history']].append(q)

    # setup camera
    hfov_rad = hfov * np.pi / 180
    vfov_rad = 2 * np.arctan(np.tan(hfov_rad / 2) * img_height / img_width)
    fx = (1.0 / np.tan(hfov_rad / 2.0)) * img_width / 2.0
    fy = (1.0 / np.tan(vfov_rad / 2.0)) * img_height / 2.0
    cx = img_width // 2
    cy = img_height // 2
    cam_intr = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

    for scene_id in questions_per_scene.keys():
        print(f"Augmenting questions for scene {scene_id}")

        if int(scene_id.split('-')[0]) < 800:
            scene_path = os.path.join(dataset_path, 'train', scene_id)
        else:
            scene_path = os.path.join(dataset_path, 'val', scene_id)
        scene_name = scene_id.split('-')[1]
        scene_mesh_dir = os.path.join(scene_path, scene_name + '.basis' + '.glb')
        navmesh_file = os.path.join(scene_path, scene_name + '.basis' + '.navmesh')

        if not os.path.exists(scene_mesh_dir):
            print(f"Scene {scene_id} does not exist")
            continue
        if not os.path.exists(navmesh_file):
            print(f"Navmesh for scene {scene_id} does not exist")
            continue

        # setup simulator
        try:
            simulator.close()
        except:
            pass

        sim_settings = {
            "scene": scene_mesh_dir,
            "default_agent": 0,
            "sensor_height": camera_height,
            "width": img_width,
            "height": img_height,
            "hfov": hfov,
            "scene_dataset_config_file": scene_dataset_config_file,
        }
        sim_config = make_cfg(sim_settings)
        try:
            simulator = habitat_sim.Simulator(sim_config)
        except:
            print(f"Failed to load scene {scene_id}")
            continue

        agent = simulator.initialize_agent(sim_settings["default_agent"])
        agent_state = habitat_sim.AgentState()

        for question_data in questions_per_scene[scene_id]:
            question = question_data['question']
            answer = question_data['answer']

            position = question_data['position']
            rotation = question_data['rotation']

            agent_state.position = position
            agent_state.rotation = rotation
            agent.set_state(agent_state)
            obs = simulator.get_sensor_observations()

            rgb = obs['color_sensor']

            save_folder = os.path.join(tempt_save_dir, f"{question_data['question_id']}_{question_data['class']}")
            os.makedirs(save_folder, exist_ok=True)

            plt.imsave(os.path.join(save_folder, 'rgb.png'), rgb)















