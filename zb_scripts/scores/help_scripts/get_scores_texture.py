import os
import json

#  现在先是 弄more ！！！

ppp = '/mmu-vcg/zb08/outputs/texture_edit/fix_final_scores/texture'
names = os.listdir(ppp)


more_json_p = '/mmu-vcg/zb08/outputs/texture_edit/train_data/rl_data/maybe_v2_final_texture/test_clean_v2.jsonl'
with open(more_json_p, 'r') as f:
    more_json = [json.loads(line) for line in f]
more_names = [os.path.basename(item['edit_image']) for item in more_json]

edge_p = ''

for name in names:
    item_path = os.path.join(ppp, name)
    with open(item_path, 'r') as f:
        cur_dict = json.load(f)
    all_s = 0
    nums = 0
    for item in cur_dict:
        img_p = item['edited_p']
        cur_name = os.path.basename(img_p)
        if cur_name not in more_names:
            # print(cur_name)
            continue
        ins_s = item['follow_instruction_score']
        if ins_s == None:
            continue
        all_s += ins_s
        nums += 1
    print(name, all_s/nums)


    print("jj")

