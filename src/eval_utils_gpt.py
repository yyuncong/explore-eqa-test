import openai
from openai import AzureOpenAI
from PIL import Image
import base64
from io import BytesIO
import os
import time
from typing import Optional
import logging

client = AzureOpenAI(
    azure_endpoint="https://chuangsweden.openai.azure.com",
    api_key=os.getenv('AZURE_OPENAI_KEY'),
    api_version="2024-02-15-preview",
)

def format_content(contents):
    formated_content = []
    for c in contents:
        formated_content.append({"type": "text", "text": c[0]})
        if len(c) == 2:
            formated_content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{c[1]}",  # yh: previously I always used jpeg format. The internet says that jpeg is smaller in size? I'm not sure.
                        "detail": "high"
                     }
                }
            )
    return formated_content
    
# send information to openai
def call_openai_api(sys_prompt, contents) -> Optional[str]:
    max_tries = 5
    retry_count = 0
    formated_content = format_content(contents)
    message_text = [
        {"role": "system", "content": sys_prompt},
        {"role": "user","content": formated_content}
    ]
    while retry_count < max_tries:
        try:
            completion = client.chat.completions.create(
                model="gpt-4o",  # model = "deployment_name"
                messages=message_text,
                temperature=0.7,
                max_tokens=4096,
                top_p=0.95,
                frequency_penalty=0,
                presence_penalty=0,
                stop=None,
            )
            return completion.choices[0].message.content
        except openai.RateLimitError as e:
            print("Rate limit error, waiting for 60s")
            time.sleep(30)
            retry_count += 1
            continue
        except Exception as e:
            print("Error: ", e)
            time.sleep(60)
            retry_count += 1
            continue

    return None

# encode tensor images to base64 format
def encode_tensor2base64(img):
    img = Image.fromarray(img)
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    buffer.seek(0)
    img_base64 = base64.b64encode(buffer.read()).decode("utf-8")
    return img_base64

def format_question(step):

    question = step["question"]
    image_goal = None
    if "task_type" in step and step["task_type"] == "image":
        with open(step["image"],"rb") as image_file:
            image_goal = base64.b64encode(image_file.read()).decode('utf-8')

    return question, image_goal

def get_step_info(step):
    # 1 get question data
    question, image_goal = format_question(step)
    # 2 get step information(egocentric, frontier, snapshot)

    # 2.1 get egocentric views
    egocentric_imgs = []
    if step.get("use_egocentric_views", False):
        for egocentric_view in step["egocentric_views"]:
            egocentric_imgs.append(encode_tensor2base64(egocentric_view))
            
    # 2.2 get frontiers
    frontier_imgs = []
    for frontier in step["frontier_imgs"]:
        frontier_imgs.append(encode_tensor2base64(frontier))
        
    # 2.3 get snapshots
    snapshot_imgs, snapshot_classes = [],[]
    obj_map = step['obj_map']
    seen_classes = set()
    for i, rgb_id in enumerate(step["snapshot_imgs"].keys()):
        snapshot_img = step["snapshot_imgs"][rgb_id]
        snapshot_imgs.append(encode_tensor2base64(snapshot_img))
        snapshot_class = [obj_map[int(sid)] for sid in step["snapshot_objects"][rgb_id]]
        # remove duplicates
        snapshot_class = sorted(list(set(snapshot_class)))
        seen_classes.update(snapshot_class)
        snapshot_classes.append(
            snapshot_class
        )
    # 2.3.3 prefiltering, note that we need the obj_id_mapping
    keep_index = list(range(len(snapshot_imgs)))
    if step.get("use_prefiltering") is True:
        n_prev_snapshot = len(snapshot_imgs)
        snapshot_classes, keep_index = prefiltering(
            question, snapshot_classes, seen_classes, step["top_k_categories"], image_goal
        )
        snapshot_imgs = [snapshot_imgs[i] for i in keep_index]
        logging.info(f"Prefiltering snapshot: {n_prev_snapshot} -> {len(snapshot_imgs)}")
    
    return question, image_goal, egocentric_imgs, frontier_imgs, snapshot_imgs, snapshot_classes, keep_index
       
def format_explore_prompt(
    question,
    egocentric_imgs,
    frontier_imgs,
    snapshot_imgs,
    snapshot_classes,
    egocentric_view = False,
    use_snapshot_class = True,
    image_goal = None
):
    sys_prompt = "Task: You are an agent in an indoor scene tasked with answering questions by observing the surroundings and exploring the environment. To answer the question, you are required to choose either a Snapshot or a Frontier image as the direction to explore.\n"
    # TODO: format interleaved text and images
    # a list of (text, image) tuples, if theres no image, use (text,)
    content = []
    # 1 here is some basic info
    #text = "Task: You are an agent in an indoor scene tasked with answering quesions by observing the surroundings and exploring the environment. To answer the question, you are required to choose either a snapshot or a frontier based on the egocentric views of your surroundings.\n"
    text = "Definitions:\n"
    text += "Snapshot: A focused observation of several objects. Choosing a snapshot means that you are selecting the observed objects in the snapshot as the target objects to help answer the question.\n"
    text += "Frontier: An unexplored region that could potentially lead to new information for answering the question. Selecting a frontier means that you will further explore that direction.\n"
    # TODO: add simple example: frontier, snapshot 
    # set | use '/n' to separate different parts
    # uppercase?
    # 2 here is the question
    # TODO: add the image goal here
    text += f"Question: {question}"
    if image_goal is not None:
        content.append((text, image_goal))
        content.append(("\n",))
    else:
        content.append((text+"\n",))
    text = "Select the Frontier/Snapshot that would help find the answer of the question.\n"
    # add the text to the content
    content.append((text,))
    # remove '/'
    if egocentric_view:
        #text += "Followings are the egocentric views of the agent (in left, right, and forward directions)\n"
        text = "The following is the egocentric view of the agent in forward direction: "
        content.append((text, egocentric_imgs[-1]))
        content.append(("\n",))
    # 3 here is the snapshot images
    text = "The followings are all the snapshots that you can explore (followed with contained object classes)\n"
    text += "Please note that the contained classes may not be accurate (wrong classes/missing classes) due to the limitation of the object detection model. "
    text += "So you still need to utilize the images to make the decision.\n"
    content.append((text,))
    if len(snapshot_imgs) == 0:
        content.append(("No Snapshot is available\n",))
    else:
        for i in range(len(snapshot_imgs)):
            content.append((f"Snapshot {i} ", snapshot_imgs[i]))
            if use_snapshot_class:
                text = ", ".join(snapshot_classes[i])
                content.append((text,))
            content.append(("\n",))
    # 4 here is the frontier images
    text = "The followings are all the Frontiers that we can explore: \n"
    content.append((text,))
    if len(frontier_imgs) == 0:
        content.append(("No Frontier is available\n",))
    else:
        for i in range(len(frontier_imgs)):
            content.append((f"Frontier {i} ", frontier_imgs[i]))
            content.append(("\n",))
    # 5 here is the format of the answer
    text = "Please provide your answer in the following format: 'Snapshot i' or 'Frontier i', where i is the index of the snapshot or frontier you choose. "
    text += "For example, if you choose the first snapshot, please type 'Snapshot 0'.\n"
    text += "You can explain the reason for your choice, but put it in a new line after the choice.\n"
    #text += "Answer: "
    content.append((text,))
    return sys_prompt, content

def format_prefiltering_prompt(
    question,
    class_list,
    top_k = 10,
    image_goal = None
):
    content = []
    sys_prompt = "You are an AI agent in a 3D indoor scene.\n"
    prompt = "Your goal is to answer questions about the scene through exploration.\n"
    prompt += "To efficiently solve the problem, you should first rank objects in the scene based on their importance.\n"
    # prompt += "You should rank the objects based on how well they can help you answer the question.\n"
    # prompt += "More important objects should be more helpful in answering the question, and should be ranked higher and first explored.\n"
    # prompt += f"Only the top {top_k} ranked objects should be included in the response.\n"
    # prompt += "If there are not enough objects, you only need to rank the objects and return all of them in ranked order.\n" 
    prompt += "These are the rules for the task.\n"
    # prompt += "RULES:\n"
    prompt += "1. Read through the whole object list.\n"
    prompt += "2. Rank objects in the list based on how well they can help your exploration given the question.\n"
    prompt += f"3. Reprint the name of all objects that may help your exploration given the question. "
    prompt += "4. Do not print any object not included in the list or include any additional information in your response.\n"
    content.append((prompt,))
    #------------------format an example-------------------------
    prompt = "Here is an example of selecting helpful objects:\n"
    # prompt += "EXAMPLE: select top 3 ranked objects\n"
    prompt += "Question: What can I use to watch my favorite shows and movies?\n"
    prompt += "Following is a list of objects that you can choose, each object one line\n"
    prompt += "painting\nspeaker\nbox\ncabinet\nlamptv\nbook rack\nsofa\noven\nbed\ncurtain\n"
    prompt += "Answer: tv\nspeaker\nsofa\nbed\n"
    content.append((prompt,))
    #------------------Task to solve----------------------------
    prompt = f"Following is the concrete content of the task and you should retrieve helpful objects in order:\n"
    prompt += f"Question: {question}"
    if image_goal is not None:
        content.append((prompt, image_goal))
        content.append(("\n",))
    else:
        content.append((prompt+"\n",))
    prompt = "Following is a list of objects that you can choose, each object one line\n"
    for i, cls in enumerate(class_list):
        prompt += f"{cls}\n"
    prompt += "Answer: "
    content.append((prompt,))
    return sys_prompt,content

def get_prefiltering_classes(
    question,
    seen_classes,
    top_k=10,
    image_goal = None
):
    prefiltering_sys,prefiltering_content = format_prefiltering_prompt(
        question, sorted(list(seen_classes)), top_k=top_k, image_goal=image_goal)
    # logging.info("Prefiltering prompt:")
    message = ''
    for c in prefiltering_content:
        message += c[0]
        if len(c) == 2:
            message += f": image {c[1][:10]}..."
    # logging.info(message)
    response = call_openai_api(prefiltering_sys, prefiltering_content)
    if response is None:
        return []
    # parse the response and return the top_k objects
    selected_classes = response.strip().split('\n')
    # logging.info(f"Prefiltering response: {selected_classes}")
    selected_classes = [cls.strip() for cls in selected_classes]
    selected_classes = [cls for cls in selected_classes if cls in seen_classes]
    selected_classes = selected_classes[:top_k]
    logging.info(f"Prefiltering selected classes: {selected_classes}")
    return selected_classes

def prefiltering(
    question,
    snapshot_classes,
    seen_classes,
    top_k=10,
    image_goal = None
):
    selected_classes = get_prefiltering_classes(
        question, seen_classes, top_k, image_goal
    )
    #print(f"Selected classes: {selected_classes}")
    keep_index = [i for i in range(len(snapshot_classes)) 
        if len(set(snapshot_classes[i]) & set(selected_classes)) > 0]
    #print("snapshot classes before filtering: ", snapshot_classes)
    snapshot_classes = [snapshot_classes[i] for i in keep_index]
    #print("snapshot classes after filtering: ", snapshot_classes)
    snapshot_classes = [sorted(list(set(s_cls)&set(selected_classes))) for s_cls in snapshot_classes]
    #print("snapshot classes after class-wise filtering",snapshot_classes)
    return snapshot_classes, keep_index
   
def explore_step(step, cfg):
    step["use_prefiltering"] = cfg.prefiltering
    step["top_k_categories"] = cfg.top_k_categories
    question, image_goal, egocentric_imgs, frontier_imgs, snapshot_imgs, snapshot_classes, snapshot_id_mapping = get_step_info(step)
    sys_prompt, content = format_explore_prompt(
        question,
        egocentric_imgs,
        frontier_imgs,
        snapshot_imgs,
        snapshot_classes,
        egocentric_view = step.get("use_egocentric_views", False),
        use_snapshot_class = True,
        image_goal = image_goal
    )
    
    # logging.info(f"Input prompt:")
    message = sys_prompt
    for c in content:
        message += c[0]
        if len(c) == 2:
            message += f"[{c[1][:10]}...]"
    # logging.info(message)

    retry_bound = 3
    final_response = None
    for _ in range(retry_bound):
        response = call_openai_api(sys_prompt, content)

        if response is None:
            print("call_openai_api returns None, retrying")
            continue

        response = response.strip()
        if "\n" in response:
            response = response.split("\n")
            response, reason = response[0], response[-1]
        else:
            reason = ""
        response = response.lower()
        try:
            choice_type, choice_id = response.split(" ")
        except Exception as e:
            print(f"Error in splitting response: {response}")
            print(e)
            continue

        response_valid = False
        if choice_type == "snapshot" and choice_id.isdigit() and 0 <= int(choice_id) < len(snapshot_imgs):
            response_valid = True
        elif choice_type == "frontier" and choice_id.isdigit() and 0 <= int(choice_id) < len(frontier_imgs):
            response_valid = True

        if response_valid:
            logging.info(f"Response: [{response}], Reason: [{reason}]")
            final_response = response
            break


    return final_response, snapshot_id_mapping
   
    
    
    