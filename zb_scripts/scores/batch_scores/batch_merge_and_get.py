import os
import json


pp_names_p = '/mmu-vcg/zb08/outputs/texture_edit/final_old_scores_final'
pp_names = os.listdir(pp_names_p)
# ddd = pp_names_p


ppp_list = []

for pp_name in pp_names:
    ddd = os.path.join(pp_names_p, pp_name)

    out_json_p = os.path.join(ddd, 'final_merged.json')
    os.remove(out_json_p)
    print(f"已删除旧文件: {out_json_p}")

    names = os.listdir(ddd)
    all_data = []
    for name in names:
        json_p = os.path.join(ddd, name)
        with open(json_p, 'r', encoding='utf-8') as f:
            data = json.load(f)
            all_data.extend(data)
    print(f"总共数据量: {len(all_data)}")


    with open(out_json_p, 'w', encoding='utf-8') as f:
        json.dump(all_data, f, ensure_ascii=False, indent=4)
    print(f"已保存至: {out_json_p}")


    total_ins = 0


    for idx, item in enumerate(all_data):
        s_ins = item['follow_instruction_score']

        if s_ins is None:
            s_ins = 0

        total_ins += s_ins

    print("Average Follow Instruction Score:", total_ins/len(data))
    print("_"*50)
