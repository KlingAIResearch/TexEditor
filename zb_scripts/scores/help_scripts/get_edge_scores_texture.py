import os
import json 

def process_texture_s(s):
    if s < 0.7:
        s = -0.2
    else:
        if s > 0.9 :
            s = 1
        else:
            s = (s - 0.7) / 0.2
    return s

def process_more_s(s):
    if s < 0.8:
        s = -0.2
    else:
        if s > 0.95 :
            s = 1
        else:
            s = (s - 0.8) / 0.15
    return s

more_json_p = '/mmu-vcg/zb08/outputs/texture_edit/train_data/rl_data/maybe_v2_final_texture/test_clean_v2.jsonl'
with open(more_json_p, 'r') as f:
    more_json = [json.loads(line) for line in f]
more_names = [os.path.basename(item['edit_image']) for item in more_json]


all_edge_scores_p = '/mmu-vcg/zb08/outputs/texture_edit/new_model_out/all_edges/score/final_merged_all.json'
with open(all_edge_scores_p, 'r') as f:
    all_edge_data = json.load(f)

all_edge_more_p = '/mmu-vcg/zb08/outputs/texture_edit/new_model_out/all_edges_115/score/final_merged_all.json'
with open(all_edge_more_p, 'r') as f:
    all_edge_more_data = json.load(f)

all_edge_data.update(all_edge_more_data)

for key in all_edge_data.keys():
    cur_dict = all_edge_data[key]
    cur_m_s = 0
    num = 0
    for name in cur_dict.keys():
        if name in more_names:
            cur_s = process_texture_s(cur_dict[name])
            cur_m_s  += cur_s
            num += 1
        else:
            pass
    print(key, cur_m_s/num)

print("hh")