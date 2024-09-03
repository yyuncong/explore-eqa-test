import os
import json

import habitat_sim
import numpy as np
import matplotlib.pyplot as plt
import random

os.environ["TRANSFORMERS_VERBOSITY"] = "error"  # disable warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HABITAT_SIM_LOG"] = (
    "quiet"  # https://aihabitat.org/docs/habitat-sim/logging.html
)
os.environ["MAGNUM_LOG"] = "quiet"


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


def main():
    # scene_dataset_config_file = '/home/hanyang/code/3d_project/explore-eqa/data/versioned_data/hm3d-0.2/hm3d/hm3d_annotated_basis.scene_dataset_config.json'
    # dataset_path = '/home/hanyang/code/3d_project/explore-eqa/data/versioned_data/hm3d-0.2/hm3d'
    # semantic_bbox_data_path = '/home/hanyang/code/3d_project/explore-eqa/data/hm3d_obj_bbox_merged'
    # goat_bench_data_dir = '/home/hanyang/code/3d_project/goatbench/data'
    scene_dataset_config_file = '/gpfs/u/home/LMCG/LMCGnngn/scratch/multisensory/MLLM/data/hm3d/hm3d_annotated_basis.scene_dataset_config.json'
    dataset_path = "/gpfs/u/home/LMCG/LMCGnngn/scratch/multisensory/MLLM/data/hm3d"
    semantic_bbox_data_path = '/gpfs/u/home/LMCG/LMCGnngn/scratch/multisensory/MLLM/data/hm3d/hm3d_obj_bbox_merged'
    goat_bench_data_dir = '/gpfs/u/home/LMCG/LMCGhazh/scratch/yanghan/goatbench/data'

    save_dir = 'goat_bench_data'
    os.makedirs(save_dir, exist_ok=True)

    random.seed(0)

    camera_height = 0
    img_width = 1280
    img_height = 1280
    hfov = 100

    # setup camera
    hfov_rad = hfov * np.pi / 180
    vfov_rad = 2 * np.arctan(np.tan(hfov_rad / 2) * img_height / img_width)
    fx = (1.0 / np.tan(hfov_rad / 2.0)) * img_width / 2.0
    fy = (1.0 / np.tan(vfov_rad / 2.0)) * img_height / 2.0
    cx = img_width // 2
    cy = img_height // 2
    cam_intr = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

    train_scenes = os.listdir(os.path.join(dataset_path, 'train'))
    val_scenes = os.listdir(os.path.join(dataset_path, 'val'))
    all_scene_ids = train_scenes + val_scenes

    for dataset_split in os.listdir(goat_bench_data_dir):
        data_save_path = os.path.join(save_dir, f"goat_bench_{dataset_split}_questions.json")
        if os.path.exists(data_save_path):
            generated_data = json.load(open(data_save_path, 'r'))
            generated_question_ids = [data['question_id'] for data in generated_data]
        else:
            generated_data = []
            generated_question_ids = []
        image_save_dir = os.path.join(save_dir, f"{dataset_split}_images")
        os.makedirs(image_save_dir, exist_ok=True)

        all_scene_data = os.listdir(os.path.join(goat_bench_data_dir, dataset_split, 'content'))
        for scene_data_path in all_scene_data:
            scene_data = json.load(open(os.path.join(goat_bench_data_dir, dataset_split, 'content', scene_data_path), 'r'))

            scene_name = scene_data_path.split('.')[0]
            scene_id = [ids for ids in all_scene_ids if scene_name in ids][0]
            if int(scene_id.split('-')[0]) < 800:
                scene_path = os.path.join(dataset_path, 'train', scene_id)
            else:
                scene_path = os.path.join(dataset_path, 'val', scene_id)
            scene_mesh_dir = os.path.join(scene_path, scene_name + '.basis' + '.glb')
            navmesh_file = os.path.join(scene_path, scene_name + '.basis' + '.navmesh')

            if not os.path.exists(scene_mesh_dir) or not os.path.exists(navmesh_file):
                print(f"Scene {scene_id} does not exist")
                continue
            if not os.path.exists(os.path.join(semantic_bbox_data_path, scene_id + '.json')):
                print(f"Scene {scene_id} does not have semantic bbox data: {os.path.join(semantic_bbox_data_path, scene_id + '.json')}")
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

            for item_id, all_items in scene_data['goals'].items():
                for item in all_items:
                    if 'lang_desc' not in item or 'image_goals' not in item or 'view_points' not in item:
                        continue

                    object_id = int(item['object_id'].split('_')[-1])
                    question_id = item_id.replace(' ', '_').replace('.basis.glb', '') + f"_{object_id}"
                    lang_desc = item['lang_desc']
                    object_coordinate = item['position']
                    object_class = item['object_category']
                    position = item['view_points'][0]['agent_state']['position']

                    if question_id in generated_question_ids:
                        continue

                    question_data = {}
                    question_data['question_id'] = question_id
                    question_data['episode_history'] = scene_id
                    question_data['category'] = 'object localization'
                    question_data['question'] = f"Could you find the object described as \'{lang_desc}\'?"
                    question_data['answer'] = object_class
                    question_data['object_id'] = object_id
                    question_data['class'] = object_class
                    question_data['position'] = position

                    # take a snapshot for the target object
                    view_pos_dict = random.choice(item['image_goals'])

                    agent_state.position = view_pos_dict['position']
                    agent_state.rotation = view_pos_dict['rotation']
                    agent.set_state(agent_state)
                    obs = simulator.get_sensor_observations()

                    rgb = obs['color_sensor']
                    plt.imsave(os.path.join(image_save_dir, f"{question_id}.png"), rgb)

                    question_data['image'] = f"{image_save_dir}/{question_id}.png"

                    generated_data.append(question_data)
                    generated_question_ids.append(question_id)

                    print(f"{dataset_split}: generated {len(generated_data)} questions")

            with open(data_save_path, 'w') as f:
                json.dump(generated_data, f, indent=4)






if __name__ == "__main__":
    main()