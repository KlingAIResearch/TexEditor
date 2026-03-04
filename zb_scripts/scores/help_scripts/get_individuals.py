import os

import json
dst_f = '/mmu-vcg/zb08/outputs/texture_edit/fix_final_scores/more'
# dst_f = '/mmu-vcg/zb08/outputs/texture_edit/fix_final_scores/texture'
os.makedirs(dst_f, exist_ok=True)
key_dict = {}
texture_json_p = '/mmu-vcg/zb08/outputs/texture_edit/final_old_scores_final/2511_dpm20_more_g3/final_merged.json'
# texture_json_p = '/mmu-vcg/zb08/outputs/texture_edit/final_old_scores_final/sftour_dpm20_texture_g3/final_merged.json'
with open(texture_json_p, 'r') as f:
    texture_data = json.load(f)

for item in texture_data:
    edited_p = item['edited_p']
    cur_key = edited_p.split('/')[-3]
    if cur_key not in key_dict:
        key_dict[cur_key] = [item]
    else:
        key_dict[cur_key].append(item)
print(key_dict.keys())

for key in key_dict:
    save_p = os.path.join(dst_f, key+'.json')
    with open(save_p, 'w') as f:
        json.dump(key_dict[key], f, indent=4)
    print("cur: ", key)
    print("len:", len(key_dict[key]))
print("jj")
