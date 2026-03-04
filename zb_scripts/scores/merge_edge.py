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

dst_f_p = '/mmu-vcg/zb08/outputs/texture_edit/new_model_out/all_edges_125_morev2/score'
# dst_f_p = '/mmu-vcg/zb08/outputs/texture_edit/new_model_out/all_edges_125/score'
# dst_f_p = '/mmu-vcg/zb08/outputs/texture_edit/new_model_out/all_edges_122/score'
# dst_f_p = '/mmu-vcg/zb08/outputs/texture_edit/new_model_out/all_edges_123/more/score'
# dst_f_p = '/mmu-vcg/zb08/outputs/texture_edit/new_model_out/all_edges_116/score'
# dst_f_p = '/mmu-vcg/zb08/outputs/texture_edit/new_model_out/all_edges_115/more/score'
# dst_f_p = '/mmu-vcg/zb08/outputs/texture_edit/new_model_out/all_edges_115/score'
names = os.listdir(dst_f_p)


# key_list = [ gemini3, gemni25, sft_base, rl, origin, qwen_11 ]
# all_data = {"rl_96":{}, "alchemist":{}}
# all_data = {"rl_gemini":{}, "rl_edge":{}}
all_data = {"rl":{},"g2":{}, "g3":{},"qwen_2509":{}}
# all_data = {"g2":{}, "g3":{},"qwen_2509":{}}
# all_data = {"rl_72":{}, "rl_64":{}, "nobase_rl":{}}
# all_data = {"gemini3":{}, "gemini2.5":{}, "sft_base":{}, "rl":{}, "origin":{}, "qwen11":{}}

for name in names:
    json_p = os.path.join(dst_f_p, name)
    with open(json_p, 'r', encoding='utf-8') as f:
        data = json.load(f)

        new_data = {}
        for item in data:
            new_data.update(item)
        for key in all_data:
            cur_data = new_data[key]
            all_data[key].update(cur_data)

print("finish merging")

# save_final_p = os.path.join(dst_f_p, 'final_merged_all.json')
# with open(save_final_p, 'w', encoding='utf-8') as f:
#     json.dump(all_data, f, ensure_ascii=False, indent=4)
# print(f"已保存至: {save_final_p}")

# for key in all_data:
#     out_json_p = os.path.join(dst_f_p, f'final_merged_{key}.json')
#     with open(out_json_p, 'w', encoding='utf-8') as f:
#         json.dump(all_data[key], f, ensure_ascii=False, indent=4)
#     print(f"已保存至: {out_json_p}")


# 下面要统计每个key下所有图像的平均分
for key in all_data:
    data = all_data[key]
    total_score = 0
    count = 0
    for img_name, score in data.items():
        if score is not None:
            total_score += process_texture_s(score) 
            # total_score += score 
            count += 1
    avg_score = total_score / count if count > 0 else 0
    print(f"Average score for {key}: {avg_score}")
print("All done")