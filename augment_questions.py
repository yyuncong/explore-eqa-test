import os
import json

from openai import AzureOpenAI
import habitat_sim
import numpy as np
import matplotlib.pyplot as plt
import base64


os.environ["TRANSFORMERS_VERBOSITY"] = "error"  # disable warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HABITAT_SIM_LOG"] = (
    "quiet"  # https://aihabitat.org/docs/habitat-sim/logging.html
)
os.environ["MAGNUM_LOG"] = "quiet"


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


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

    client = AzureOpenAI(
        azure_endpoint='https://yuncong.openai.azure.com',
        api_key=os.getenv('AZURE_OPENAI_KEY'),
        api_version='2024-02-15-preview',
    )

    sys_prompt = "You are an intelligent AI agent."
    bg_prompt = "I have a question about an object at the center of an image, which I will provide with you. " + \
           "The object the question refers to might be vague, since an object can have multiple synonyms. " + \
           "So I need to augment the question to increase its variety for better training a model. " + \
           "You need to paraphrase the question by replacing the name of that object with its synonym based on your observation.\n" + \
           "For example\n" + \
           "Question: Is there a cushion on the armchair in front of the window in the living room?\n" + \
           "Paraphrased Question: Is there a pillow on the sofa chair in front of the window in the living room?\n" + \
           "Question: What is the primary color of the desk chair that is in front of the white desk with shelves and various items on it?\n" + \
           "Paraphrased Question: What is the main color of the office chair that is in front of the white desk with shelves and various items on it?\n" + \
           "Also, in some case, there might be no synonym for the object. In that case, you can just return the same question. For example:\n" + \
           "Question: Where is the white printer located?\n" + \
           "Paraphrased Question: Where is the white printer located?\n" + \
           "Question: Is the light around the mirror in the bathroom turned on?\n" + \
           "Paraphrased Question: Is the light around the mirror in the bathroom turned on?\n" + \
           "Also, there is another case where the question doesn't explicitly refer to an object. In that case, you can just return the same question. For example:\n" + \
           "Question: What can I use to quickly heat up my food?\n" + \
           "Paraphrased Question: What can I use to quickly heat up my food?\n\n" + \
           "Now its your turn. Note that you just need to return the paraphrased question, and don't need to give any explanations.\n"


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

        for question_data in questions_per_scene[scene_id][:3]:
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

            # prompt gpt
            content = [{"type": "text", "text": bg_prompt}]

            content += [{"type": "text", "text": f"Here is the image for the question: "}]
            content += [{
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{encode_image(os.path.join(save_folder, 'rgb.png'))}",
                    "detail": "high",
                }
            }]

            content += [{"type": "text", "text": f"Question: {question}\n"}]
            content += [{"type": "text", "text": "Paraphrased Question: "}]

            try:
                response = client.chat.completions.create(
                    model='gpt-4o',
                    messages=[
                        {"role": "system", "content": [{"type": "text", "text": sys_prompt}]},
                        {"role": "user", "content": content},
                    ],
                    max_tokens=200,
                    seed=42,
                    temperature=0.2
                )

                output = response.choices[0].message.content
            except Exception as e:
                print(f"Failed to generate question for question {question_data['question_id']}")
                print(e)
                continue

            print(f"Question: {question}")
            print(f"Paraphrased Question: {output}")















