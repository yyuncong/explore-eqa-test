import os
import pickle
from typing import Dict
import json
from tqdm import tqdm

from openai import AzureOpenAI
import base64

from src.tsdf_new_cg import SnapShot

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def main():
    eval_data_path = 'data/open-eqa-v0.json'
    scene_data_dir = '/project/pi_chuangg_umass_edu/yuncong/scene_data_scannet/'
    result_save_path = 'emeqa_scannet_eval_results.json'

    sys_prompt = "You are an intelligent agent."
    prompt = f"I have a collection of observations of an indoor scene. " \
             f"The observations cover the entire scene and each observation focuses on a different part of the scene. " \
             f"Based on the observations, you need to answer a question about the scene. " \
             f"Note that you need to directly answer the question after the prompt \"Answer: \", and you do not need to provide any further reasoning.\n" \

    # load eval data
    eval_data = json.load(open(eval_data_path, 'r'))
    scene_id_to_questions = {}
    for question_data in eval_data:
        if 'scannet-v0' not in question_data['episode_history']:
            continue
        scene_id = question_data['episode_history'].split('/')[-1]
        if scene_id not in scene_id_to_questions:
            scene_id_to_questions[scene_id] = []
        scene_id_to_questions[scene_id].append(question_data)

    client = AzureOpenAI(
        azure_endpoint='https://yuncong.openai.azure.com',
        api_key=os.getenv('AZURE_OPENAI_KEY'),
        api_version='2024-02-15-preview',
    )

    evaluation_results: Dict[str, str] = {}  # scene_id -> gpt answer

    for scene_index, scene_folder in enumerate(os.listdir(scene_data_dir)):
        print(f"Processing scene {scene_folder} ({scene_index}/{len(os.listdir(scene_data_dir))})")

        scene_folder_path = os.path.join(scene_data_dir, scene_folder)
        scene_graph_data_path = os.path.join(scene_folder_path, 'snapshots_inclusive_merged.pkl')
        scene_graph_data: Dict[str, SnapShot] = pickle.load(open(scene_graph_data_path, 'rb'))

        all_snapshot_paths = [os.path.join(scene_folder_path, 'results', snapshot_id) for snapshot_id in scene_graph_data.keys()]
        all_snapshots_base64 = [encode_image(snapshot_path) for snapshot_path in all_snapshot_paths]

        scene_id = scene_folder
        for question_data in tqdm(scene_id_to_questions[scene_id]):
            question = question_data['question']
            question_id = question_data['question_id']

            content = [{"type": "text", "text": prompt}]
            content.append({"type": "text", "text": f"Question: {question}\n"})
            content.append({"type": "text", "text": f"The followings are the observations of the scene:\n"})

            for image_index, snapshot_base64 in enumerate(all_snapshots_base64):
                content.append({"type": "text", "text": f"Image {image_index} "})
                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{snapshot_base64}",
                        "detail": "high"
                    }
                })

            content.append({"type": "text", "text": f"Answer: "})

            try:
                response = client.chat.completions.create(
                    model='gpt-4o',
                    messages=[
                        {"role": "system", "content": [{"type": "text", "text": sys_prompt}]},
                        {"role": "user", "content": content},
                    ],
                    max_tokens=500,
                    seed=42,
                    temperature=0.2
                )
                output = response.choices[0].message.content
                evaluation_results[question_id] = output
            except Exception as e:
                print(f"Error processing scene {scene_id}, question {question_id}: {e}")
                continue

            # save intermediate results
            json.dump(evaluation_results, open(result_save_path, 'w'))


if __name__ == '__main__':
    main()