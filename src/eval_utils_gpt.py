from src.keys import api_key
import openai
from openai import AzureOpenAI
from PIL import Image
import base64
from io import BytesIO

client = AzureOpenAI(
    azure_endpoint="https://yuncong.openai.azure.com/",
    api_key=api_key,
    api_version="2024-02-15-preview",
)

def format_content(contents):
    formated_content = []
    for c in contents:
        formated_content.append({"type": "text", "text": c[0]})
        if len(c) == 2:
            formated_content.append(
                {"type": "image_url", 
                 "image_url": {
                     "url": f"data:image/png;base64,{c[1]}"
                     }
                 }
            )
    return formated_content
    
# send information to openai
def call_openai_api(client, sys_prompt, contents):
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
            break
        except openai.RateLimitError as e:
            print("Rate limit error, waiting for 60s")
            time.sleep(60)
            retry_count += 1
            continue
        except Exception as e:
            print("Error: ", e)
            exit(0)
    # print('openai api response {}'.format(completion))
    return completion.choices[0].message.content

# encode tensor images to base64 format
def encode_tensor2base64(img):
    img = Image.fromarray(img)
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    buffer.seek(0)
    img_base64 = base64.b64encode(buffer.read()).decode("utf-8")
    return img_base64

def get_step_info(step):
    # 1 get question data
    question = step['question']
    # 2 get step information(egocentric, frontier, snapshot)
    
    # 2.1 get egocentric views
    egocentric_imgs = []
    if step.get("use_egocentric_view", True):
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
        snapshot_classes, keep_index = prefiltering(
            question, snapshot_classes, seen_classes, step["top_k_categories"]
        )
        snapshot_imgs = [snapshot_imgs[i] for i in keep_index]
    
    return question, egocentric_imgs, frontier_imgs, snapshot_imgs, snapshot_classes, keep_index
       
def format_explore_prompt(
    question,
    egocentric_imgs,
    frontier_imgs,
    snapshot_imgs,
    snapshot_classes,
    egocentric_view = False,
    use_snapshot_class = True
):
    sys_prompt = "Task: You are an agent in an indoor scene tasked with answering quesions by observing the surroundings and exploring the environment. To answer the question, you are required to choose either a snapshot or a frontier based on the egocentric views of your surroundings.\n"
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
    text += f"Question:{question}\n"
    text += "Select the frontier/snapshot that would help find the answer of the question.\n"
    # add the text to the content
    content.append((text,))
    # TODO: only use 1 egocentric view
    # TODO: see the error type | regenerate the answer if the format is incorrect
    # TODO: direct ask for topk
    # remove '/'
    if egocentric_view:
        #text += "Followings are the egocentric views of the agent (in left, right, and forward directions)\n"
        text = "Followings is the egocentric view of the agent (in forward direction)"
        content.append((text, egocentric_imgs[-1]))
        content.append(("\n",))
    # 3 here is the snapshot images
    text = "Below are all the snapshots that we can explore (followed with contained object classes)\n"
    text += "Please note that the contained class may not be accurate(wrong classes/missing classes) due to the limitation of the object detection model.\n"
    text += "So you still need to utilize the images to make the decision.\n"
    content.append((text,))
    if len(snapshot_imgs) == 0:
        content.append(("No snapshot is available\n",))
    else:
        for i in range(len(snapshot_imgs)):
            content.append((f"SNAPSHOT {i} ", snapshot_imgs[i]))
            if use_snapshot_class:
                text = ", ".join(snapshot_classes[i])
                content.append((text,))
            content.append(("\n",))
    # 4 here is the frontier images
    text = "Below are all the frontiers that we can explore\n"
    content.append((text,))
    if len(frontier_imgs) == 0:
        content.append(("No frontier is available\n",))
    else:
        for i in range(len(frontier_imgs)):
            content.append((f"FRONTIER {i} ", frontier_imgs[i]))
            content.append(("\n",))
    # 5 here is the format of the answer
    text = "Please provide your answer in the following format: 'snapshot i' or 'frontier i', where i is the index of the snapshot or frontier you choose."
    text += "For example, if you choose the first snapshot, please type 'snapshot 0'."
    text += "You can explain the reason for your choice, but put it in a new line after the choice."
    content.append((text,))
    return sys_prompt, content

def format_prefiltering_prompt(
    question,
    class_list,
    top_k = 10
):
    sys_prompt = "You are an object selector, part of an AI agent in a 3D indoor scene.\n"
    prompt = "The goal of the AI agent is to answer questions about the scene through exploration.\n"
    prompt += "In order to efficiently solve the problem, you should rank objects in the scene based on their importance.\n"
    prompt += "More important objects should be more helpful in answering the question, and should be ranked higher and first explored.\n"
    prompt += f"Only the top {top_k} ranked objects should be included in the response.\n"
    prompt += "If there are not enough objects, you only need to rank the objects and return all of them in ranked order.\n" 
    prompt += "Following is the rules for the task.\n"
    prompt += "RULES:\n"
    prompt += "1. Read through the whole object list.\n"
    prompt += "2. Rank objects in the list based on how well they can help you answer the question.\n"
    prompt += f"3. Reprint the name of top {top_k} objects. "
    prompt += "If there are not enough objects, reprint all of them in ranked order. Each object should be printed on a new line.\n"
    prompt += "4. Do not print any object not included in the list or include any additional information in your response.\n"
    #------------------format an example-------------------------
    prompt += "Here is an example"
    prompt += "EXAMPLE: select top 3 ranked objects\n"
    prompt += "Given question: What can I use to watch my favorite shows and movies?"
    prompt += "Following is a list of objects that you can choose, each object one line\n"
    prompt += "painting\nspeaker\nbox\ncabinet\nlamp\ncouch\npillow\ncabinet\ntv\nbook rack\nwall panel\npainting\nstool\ntv stand\n"
    prompt += "Your answer should be:tv\ntv stand\nspeaker"
    #------------------Task to solve----------------------------
    prompt += "Following is the concrete content of the task\n"
    prompt += f"Given question: {question}\n"
    prompt += "Following is a list of objects that you can choose, each object one line\n"
    for i, cls in enumerate(class_list):
        prompt += f"{cls}\n"
    return sys_prompt,[(prompt,)]

def get_prefiltering_classes(
    question,
    seen_classes,
    top_k=10
): 
    prefiltering_sys,prefiltering_content = format_prefiltering_prompt(
        question, sorted(list(seen_classes)))
    response = call_openai_api(client, prefiltering_sys,prefiltering_content)
    # parse the response and return the top_k objects
    selected_classes = response.split('\n')
    selected_classes = [cls for cls in selected_classes if cls in seen_classes]
    selected_classes = selected_classes[:top_k]
    return selected_classes

def prefiltering(
    question,
    snapshot_classes,
    seen_classes,
    top_k=10
):
    selected_classes = get_prefiltering_classes(
        question, seen_classes, top_k
    )
    print(f"Selected classes: {selected_classes}")
    keep_index = [i for i in range(len(snapshot_classes)) 
        if len(set(snapshot_classes[i]) & set(selected_classes)) > 0]
    print("snapshot classes before filtering: ", snapshot_classes)
    snapshot_classes = [snapshot_classes[i] for i in keep_index]
    print("snapshot classes after filtering: ", snapshot_classes)
    snapshot_classes = [sorted(list(set(s_cls)&ranking)) for s_cls in snapshot_classes]
    print("snapshot classes after class-wise filtering",snapshot_classes)
    return snapshot_classes, keep_index
   
def explore_step(step, cfg):
    step["use_prefiltering"] = cfg.prefiltering
    step["top_k_categories"] = cfg.top_k_categories
    question, egocentric_imgs, frontier_imgs, snapshot_imgs, snapshot_classes, snapshot_id_mapping = get_step_info(step)
    sys_prompt, content = format_explore_prompt(
        question,
        egocentric_imgs,
        frontier_imgs,
        snapshot_imgs,
        snapshot_classes,
        egocentric_view = step.get("use_egocentric_view", False),
        use_snapshot_class = True
    )
    
    print(f"the size of frontier is {len(frontier_imgs)}")
    print(f"the input prompt:\n {sys_prompt + ''.join([c[0] for c in content])}")
    
    # add retry mechanism to avoid invalid answer format
    valid_response, retry_bound = False, 3
    while not valid_response and retry_bound > 0:
        response = call_openai_api(client, sys_prompt, content)
        if "\n" in response:
            response = response.split("\n")
            response, reason = response[0], response[-1]
        choice_type, choice_id = response.split(" ")
        if choice_type == "snapshot" and 0 <= int(choice_id) < len(snapshot_imgs):
            valid_response = True
        elif choice_type == "frontier" and 0 <= int(choice_id) < len(frontier_imgs):
            valid_response = True
        else:
            retry_bound -= 1
        if valid_response:
            print(f"the reason for {response} is \n {reason}")
    return response, snapshot_id_mapping
   
    
    
    