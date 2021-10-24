import torch
from pytorch_i3d import InceptionI3d

weights = '../../archived/asl2000/FINAL_nslt_2000_iters=5104_top1=32.48_top5=57.31_top10=66.31.pt'
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

i3d = InceptionI3d(400, in_channels=3)
i3d.replace_logits(num_classes = 2000)
i3d.load_state_dict(torch.load(weights,map_location=torch.device('cpu')))  # nslt_2000_000700.pt nslt_1000_010800 nslt_300_005100.pt(best_results)  nslt_300_005500.pt(results_reported) nslt_2000_011400
i3d.to(device)
i3d.eval()