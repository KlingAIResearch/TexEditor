import os
import json 

#  现在先是 弄more ！！！



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

ppp = '/mmu-vcg/zb08/outputs/texture_edit/fix_final_scores/more'
names = os.listdir(ppp)
all_gemini_s = []

all_edge_scores_p = '/mmu-vcg/zb08/outputs/texture_edit/new_model_out/all_edges/more_score/final_merged_all.json'
with open(all_edge_scores_p, 'r') as f:
    all_edge_data = json.load(f)

all_edge_more_p = '/mmu-vcg/zb08/outputs/texture_edit/new_model_out/all_edges_115/more/score/final_merged_all.json'
with open(all_edge_more_p, 'r') as f:
    all_edge_more_data = json.load(f)

all_edge_data.update(all_edge_more_data)
# 这两个分 我都不太想带rl   但还是带吧  把 rl 72 的 edge 带进来 

# 我真正想比的 其实就是 qwen11 g3 rl72

new_all = {}
for name in names:
    cur_dict = {}
    mmm_name = None
    if 'rl' in name or '2511' in name or 'gemini-3' in name: 
        if 'rl_72' in name:
            mmm_name = 'rl_72'
        if '2511' in name:
            mmm_name = 'qwen11'
        if 'gemini-3' in name:
            mmm_name = 'gemini3'
        print(name)
    else:
        continue

    path = os.path.join(ppp, name)
    with open(path, 'r') as f:
        data = json.load(f)

    cur_edge_data = all_edge_data[mmm_name]

    for item in data:
        cur_name =  os.path.basename(item['edited_p']) 
        cur_edge_s = cur_edge_data[cur_name]
        item['edge_s'] = cur_edge_s
        item['edge_s_normal'] = process_more_s(cur_edge_s)
        ins_s = item['follow_instruction_score']/10 if item['follow_instruction_score'] else 0.0 
        item['combine_11'] = item['edge_s_normal'] * 0.4 + ins_s * 0.6
        check_score = ins_s * 0.2 + item['edge_s_normal'] *0.8
        item['combine_check'] = ins_s if item['edge_s_normal'] > 0.6 else check_score
        print("hh")
        cur_dict[cur_name] = item
    new_all[mmm_name] = cur_dict

save_f = '/mmu-vcg/zb08/outputs/texture_edit/aa_4_show'
save_name = 'show_eval.json'
with open(os.path.join(save_f, save_name), 'w') as f:
    json.dump(new_all, f, indent=4)
print("jj")

