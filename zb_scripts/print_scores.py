import json
# 读取jsonl

ppp = '/mmu-vcg/zb08/outputs/texture_edit/final_scores/sft_dpm20_final_v2_texture_edge_5050.jsonl'
# ppp = '/mmu-vcg/zb08/outputs/texture_edit/final_scores/orgin_dpm20_final_v2_texture_edge_5050.jsonl'
with open(ppp, "r", encoding="utf-8") as f:
    metadatas = [json.loads(line) for line in f]

gemini_scores = 0
sam_scores = 0
for item in metadatas:
    gemini_scores += item['zb_gemini']
    sam_scores += item['sam_iou']

print(f"gemini_scores: {gemini_scores/len(metadatas)}")
print(f"sam_scores: {sam_scores/len(metadatas)}")
